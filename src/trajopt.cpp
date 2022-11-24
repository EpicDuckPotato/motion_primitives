#include "trajopt.h"

#include <pinocchio/spatial/explog.hpp>

Matrix3d quat_diff_to_rmat(const Ref<const Vector4d>& q0, const Ref<const Vector4d>& q1) {
  return Quaterniond(q0).toRotationMatrix().transpose()*Quaterniond(q1).toRotationMatrix();
}

Vector3d log_quat_diff(const Ref<const Vector4d>& q0, const Ref<const Vector4d>& q1) {
  return pin::log3(quat_diff_to_rmat(q0, q1));
}

Trajopt::Trajopt(boost::shared_ptr<pin::Model> pin_model, boost::shared_ptr<pin::GeometryModel> pin_geom) : pin_model(pin_model), pin_geom(pin_geom)  {
  solver = boost::make_shared<OsqpEigen::Solver>();
  solver->settings()->setVerbosity(false);
  solver->settings()->setWarmStart(true);
  solver->settings()->setAdaptiveRhoInterval(50);

  distances = VectorXd::Zero(pin_geom->collisionPairs.size());
  distance_jacobian = MatrixXd::Zero(pin_geom->collisionPairs.size(), pin_model->nv);

  pin_data = pin::Data(*pin_model);
  geom_data = pin::GeometryData(*pin_geom);

  for (int i = 0; i < pin_model->njoints; ++i) {
    joint_jacobians.push_back(MatrixXd::Zero(6, pin_model->nv));
  }
}

bool Trajopt::optimize(std::vector<VectorXd>& q_trj, const std::vector<VectorXd>& q_ref, const Ref<const VectorXd> &qstart, const Ref<const Vector3d> &goal_pos, double dt, double q_cost, double v_cost, double vdot_cost) {
  int steps = q_ref.size();
  double dt2 = dt*dt;
  int num_rotary = pin_model->nv - 6;

  int num_contact_pairs = pin_geom->collisionPairs.size();

  q_trj = q_ref;
  q_trj[0] = qstart;
  q_trj.back().head<3>() = goal_pos;

  // Calculate reference velocities and accelerations
  vector<VectorXd> v_ref(steps - 1);
  vector<VectorXd> vdot_ref(steps - 2);
  for (int step = 0; step < steps - 1; ++step) {
    v_ref[step] = VectorXd::Zero(pin_model->nv);
    v_ref[step].head<3>() = (q_ref[step + 1].head<3>() - q_ref[step].head<3>())/dt;
    v_ref[step].segment<3>(3) = log_quat_diff(q_ref[step].segment<4>(3), q_ref[step + 1].segment<4>(3))/dt;
    v_ref[step].tail(num_rotary) = (q_ref[step + 1].tail(num_rotary) - q_ref[step].tail(num_rotary))/dt;
  }
  for (int step = 0; step < steps - 2; ++step) {
    vdot_ref[step] = (v_ref[step + 1] - v_ref[step])/dt;
  }

  int num_decision_vars = pin_model->nv*steps;
  int num_constraints = 3 + pin_model->nv + num_contact_pairs*steps; // Goal constraint, start constraint, and non-penetration constraints

  VectorXd gradient(num_decision_vars);
  SparseMatrix<double> hessian(num_decision_vars, num_decision_vars);
  SparseMatrix<double> linearMatrix(num_constraints, num_decision_vars);
  vector<Triplet<double>> hessian_triplets;
  vector<Triplet<double>> linearMatrix_triplets;
  VectorXd lowerBound(num_constraints);
  VectorXd upperBound(num_constraints);

  // q cost stuff
  VectorXd qdiff = VectorXd::Zero(pin_model->nv);
  MatrixXd dqdiff = MatrixXd::Identity(pin_model->nv, pin_model->nv);
  MatrixXd qdiff_hess = MatrixXd::Identity(pin_model->nv, pin_model->nv);

  // v cost stuff
  VectorXd v = VectorXd::Zero(pin_model->nv);
  MatrixXd dv = -MatrixXd::Identity(pin_model->nv, 2*pin_model->nv)/dt;
  dv.rightCols(pin_model->nv) = MatrixXd::Identity(pin_model->nv, pin_model->nv)/dt;
  MatrixXd v_hess = dv.transpose()*dv;

  // vdot cost stuff
  VectorXd v_next = VectorXd::Zero(pin_model->nv);
  MatrixXd dv_next = MatrixXd::Zero(pin_model->nv, 2*pin_model->nv);

  VectorXd vdot = VectorXd::Zero(pin_model->nv);
  MatrixXd dvdot = MatrixXd::Identity(pin_model->nv, 3*pin_model->nv)/dt2;
  dvdot.block(0, pin_model->nv, pin_model->nv, pin_model->nv) = -2*MatrixXd::Identity(pin_model->nv, pin_model->nv)/dt2;
  dvdot.rightCols(pin_model->nv) = MatrixXd::Identity(pin_model->nv, pin_model->nv)/dt2;
  MatrixXd vdot_hess = dvdot.transpose()*dvdot;

  // FCL stuff
  Vector3d normal;
  Vector3d t0;
  Matrix<double, 3, 6> adj0 = Matrix<double, 3, 6>::Identity();
  Vector3d t1;
  Matrix<double, 3, 6> adj1 = Matrix<double, 3, 6>::Identity();

  Matrix3d rmat_diff;

  int max_outer = 1;
  for (int outer = 0; outer < max_outer; ++outer) {
    // TODO: how much stuff do I actually have to clear?
    solver->clearSolver();
    solver->data()->clearHessianMatrix();
    solver->data()->clearLinearConstraintsMatrix();

    // TODO: probably don't need to do this, right?
    solver->data()->setNumberOfVariables(num_decision_vars);
    solver->data()->setNumberOfConstraints(num_constraints);

    gradient.setZero();
    linearMatrix_triplets.clear();
    hessian_triplets.clear();
    lowerBound.setZero();
    upperBound.setZero();

    // Goal constraint
    for (int i = 0; i < 3; ++i) {
      linearMatrix_triplets.push_back(Triplet<double>(i, pin_model->nv*(steps - 1) + i, 1));
    }

    // Start constraint
    for (int i = 0; i < pin_model->nv; ++i) {
      linearMatrix_triplets.push_back(Triplet<double>(3 + i, i, 1));
    }
    
    for (int step = steps - 1; step >= 0; --step) {
      int startrow = step*pin_model->nv;
      int startcol = startrow;

      // Configuration deviation penalty
      qdiff.head<3>() = q_trj[step].head<3>() - q_ref[step].head<3>();
      qdiff.tail(num_rotary) = q_trj[step].tail(num_rotary) - q_ref[step].tail(num_rotary);
      rmat_diff = quat_diff_to_rmat(q_ref[step].segment<4>(3), q_trj[step].segment<4>(3));
      qdiff.segment<3>(3) = pin::log3(rmat_diff);
      pin::Jlog3(rmat_diff, dqdiff.block<3, 3>(3, 3));

      // Gradient wrt configuration deviation
      gradient.segment(startrow, pin_model->nv) += q_cost*dqdiff.transpose()*qdiff;
      
      // Hessian wrt configuration deviation
      qdiff_hess = dqdiff.transpose()*dqdiff; 
      for (int i = 0; i < pin_model->nv; ++i) {
        for (int j = 0; j < pin_model->nv; ++j) {
          hessian_triplets.push_back(Triplet<double>(startrow + i, startrow + j, qdiff_hess(i, j)));
        }
      }

      if (step < steps - 1) {
        // Velocity deviation penalty
        v.head<3>() = (q_trj[step + 1].head<3>() - q_trj[step].head<3>())/dt;
        v.tail(num_rotary) = (q_trj[step + 1].tail(num_rotary) - q_trj[step].tail(num_rotary))/dt;
        rmat_diff = quat_diff_to_rmat(q_trj[step].segment<4>(3), q_trj[step + 1].segment<4>(3));
        v.segment<3>(3) = pin::log3(rmat_diff)/dt;
        pin::Jlog3(rmat_diff, dv.block<3, 3>(3, pin_model->nv + 3));
        dv.block<3, 3>(3, pin_model->nv + 3) /= dt;
        dv.block<3, 3>(3, 3) = -dv.block<3, 3>(3, pin_model->nv + 3)*rmat_diff.transpose();

        // Gradient
        gradient.segment(startrow, 2*pin_model->nv) += v_cost*dv.transpose()*(v - v_ref[step]);

        // Hessian
        v_hess = v_cost*dv.transpose()*dv;
        for (int i = 0; i < 2*pin_model->nv; ++i) {
          for (int j = 0; j < 2*pin_model->nv; ++j) {
            hessian_triplets.push_back(Triplet<double>(startrow + i, startrow + j, v_hess(i, j)));
          }
        }
      }

      if (step < steps - 2) {
        // Acceleration deviation penalty
        vdot = (v_next - v)/dt;

        dvdot.leftCols(pin_model->nv) = -dv.leftCols(pin_model->nv)/dt;
        dvdot.block(0, pin_model->nv, pin_model->nv, pin_model->nv) = (dv_next.leftCols(pin_model->nv) - dv.rightCols(pin_model->nv))/dt;
        dvdot.rightCols(pin_model->nv) = dv_next.rightCols(pin_model->nv)/dt;

        // Gradient
        gradient.segment(startrow, 3*pin_model->nv) += vdot_cost*dvdot.transpose()*(vdot - vdot_ref[step]);

        // Hessian
        vdot_hess = vdot_cost*dvdot.transpose()*dvdot;
        for (int i = 0; i < 3*pin_model->nv; ++i) {
          for (int j = 0; j < 3*pin_model->nv; ++j) {
            hessian_triplets.push_back(Triplet<double>(startrow + i, startrow + j, vdot_hess(i, j)));
          }
        }
      }

      v_next = v;
      dv_next = dv;

      // Penetration constraints
      if (num_contact_pairs) {
        int cidx = 3 + pin_model->nv + num_contact_pairs*step;
        pin::forwardKinematics(*pin_model, pin_data, q_trj[step]);
        pin::computeJointJacobians(*pin_model, pin_data);
        for (int j = 0; j < pin_model->njoints; ++j) {
          pin::getJointJacobian(*pin_model, pin_data, j, pin::ReferenceFrame::LOCAL_WORLD_ALIGNED, joint_jacobians[j]);
        }
        pin::computeDistances(*pin_model, pin_data, *pin_geom, geom_data);

        for (int p = 0; p < pin_geom->collisionPairs.size(); ++p) {
          lowerBound(cidx + p) = -geom_data.distanceResults[p].min_distance;
          upperBound(cidx + p) = 10000;
          if (geom_data.distanceResults[p].min_distance < 0) {
            normal = geom_data.distanceResults[p].normal;
          } else {
            normal = geom_data.distanceResults[p].nearest_points[1] - geom_data.distanceResults[p].nearest_points[0];
          }
          normal.normalize();

          int jidx0 = pin_geom->geometryObjects[pin_geom->collisionPairs[p].first].parentJoint;
          t0 = geom_data.distanceResults[p].nearest_points[0] - pin_data.oMi[jidx0].translation();
          adj0.rightCols<3>() = -pin::skew(t0);

          int jidx1 = pin_geom->geometryObjects[pin_geom->collisionPairs[p].second].parentJoint;
          t1 = geom_data.distanceResults[p].nearest_points[1] - pin_data.oMi[jidx1].translation();
          adj1.rightCols<3>() = -pin::skew(t1);

          distance_jacobian.row(p) = normal.transpose()*(adj1*joint_jacobians[jidx1] - adj0*joint_jacobians[jidx0]);
        }

        distance_jacobian.leftCols<3>() = distance_jacobian.leftCols<3>()*Quaterniond(q_trj[step].segment<4>(3)).toRotationMatrix().transpose();
        for (int i = 0; i < num_contact_pairs; ++i) {
          for (int j = 0; j < pin_model->nv; ++j) {
            linearMatrix_triplets.push_back(Triplet<double>(cidx + i, startrow + j, distance_jacobian(i, j)));
          }
        }
      }
    }

    hessian.setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
    linearMatrix.setFromTriplets(linearMatrix_triplets.begin(), linearMatrix_triplets.end());

    if(!solver->data()->setHessianMatrix(hessian)) return false;
    if(!solver->data()->setGradient(gradient)) return false;
    if(!solver->data()->setLinearConstraintsMatrix(linearMatrix)) return false;
    if(!solver->data()->setLowerBound(lowerBound)) return false;
    if(!solver->data()->setUpperBound(upperBound)) return false;

    if(!solver->initSolver()) return false;

    // Solve the QP problem
    bool solved = solver->solve();
    if (!solved) {
      cout << "status: " << solver->workspace()->info->status_val << endl;
      return false;
    }

    VectorXd primal = solver->getSolution();

    /*
    std::cout << (linearMatrix*primal - lowerBound).minCoeff() << std::endl;
    std::cout << (linearMatrix*primal - upperBound).maxCoeff() << std::endl;
    std::cout << 0.5*primal.transpose()*hessian*primal + gradient.transpose()*primal << std::endl;
    */

    for (int step = 0; step < steps; ++step) {
      q_trj[step].head<3>() += primal.segment<3>(step*pin_model->nv);
      q_trj[step].tail(num_rotary) += primal.segment(step*pin_model->nv + 6, num_rotary);
      Quaterniond quat_plus = Quaterniond(q_trj[step].segment<4>(3))*Quaterniond(pin::exp3(primal.segment<3>(step*pin_model->nv + 3)));
      q_trj[step](3) = quat_plus.x();
      q_trj[step](4) = quat_plus.y();
      q_trj[step](5) = quat_plus.z();
      q_trj[step](6) = quat_plus.w();

      // primal.segment(step*pin_model->nv + 6, num_rotary).setZero();
    }

    /*
    std::cout << (linearMatrix*primal - lowerBound).minCoeff() << std::endl;
    std::cout << (linearMatrix*primal - upperBound).maxCoeff() << std::endl;
    std::cout << 0.5*primal.transpose()*hessian*primal + gradient.transpose()*primal << std::endl;
    */
  }
  return true;
}
