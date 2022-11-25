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

bool Trajopt::optimize(std::vector<VectorXd>& q_trj, const std::vector<VectorXd>& q_ref, const Ref<const VectorXd> &qstart, const Ref<const Vector3d> &goal_pos, double dt, double q_cost, double v_cost, double vdot_cost, int max_outer) {
  int steps = q_ref.size();
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
    pin::difference(*pin_model, q_ref[step], q_ref[step + 1], v_ref[step]);
    v_ref[step] /= dt;
  }
  for (int step = 0; step < steps - 2; ++step) {
    vdot_ref[step] = (v_ref[step + 1] - v_ref[step])/dt;
  }

  // Changes in configuration, velocity, and acceleration
  int vars_per_step = 3*pin_model->nv;
  int num_decision_vars = vars_per_step*steps - 3*pin_model->nv; // No acceleration for last two steps, no velocity for last step
  int num_constraints = 3 + pin_model->nv + num_contact_pairs*steps + pin_model->nv*(2*steps - 3); // Goal constraint, start constraint, non-penetration constraints, velocity equality constraints, acceleration equality constraints

  VectorXd gradient(num_decision_vars);
  SparseMatrix<double> hessian(num_decision_vars, num_decision_vars);
  SparseMatrix<double> linearMatrix(num_constraints, num_decision_vars);
  vector<Triplet<double>> hessian_triplets;
  vector<Triplet<double>> linearMatrix_triplets;
  VectorXd lowerBound(num_constraints);
  VectorXd upperBound(num_constraints);

  // q cost stuff
  VectorXd qdiff(pin_model->nv);
  MatrixXd dqdiff(pin_model->nv, pin_model->nv);
  MatrixXd qdiff_hess(pin_model->nv, pin_model->nv);

  // v cost stuff
  VectorXd v(pin_model->nv);
  MatrixXd dv(pin_model->nv, 2*pin_model->nv);

  // vdot cost stuff
  VectorXd v_next(pin_model->nv);
  MatrixXd dv_next(pin_model->nv, 2*pin_model->nv);

  VectorXd vdot(pin_model->nv);

  // FCL stuff
  Vector3d normal;
  Vector3d t0;
  Matrix<double, 3, 6> adj0 = Matrix<double, 3, 6>::Identity();
  Vector3d t1;
  Matrix<double, 3, 6> adj1 = Matrix<double, 3, 6>::Identity();

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

    int cidx = 0;

    // Goal constraint
    int var_idx = vars_per_step*(steps - 2) + 2*pin_model->nv;
    for (int i = 0; i < 3; ++i) {
      linearMatrix_triplets.push_back(Triplet<double>(cidx, var_idx, 1));
      ++var_idx;
      ++cidx;
    }

    // Start constraint
    for (int i = 0; i < pin_model->nv; ++i) {
      linearMatrix_triplets.push_back(Triplet<double>(cidx, i, 1));
      ++cidx;
    }
    
    for (int step = steps - 1; step >= 0; --step) {
      int var_idx = vars_per_step*step;
      if (step == steps - 1) {
        var_idx -= pin_model->nv;
      }

      // Configuration deviation penalty
      pin::difference(*pin_model, q_ref[step], q_trj[step], qdiff);
      pin::dDifference(*pin_model, q_ref[step], q_trj[step], dqdiff, pin::ArgumentPosition::ARG1);

      // Gradient wrt configuration deviation
      gradient.segment(var_idx, pin_model->nv) += q_cost*dqdiff.transpose()*qdiff;
      
      // Hessian wrt configuration deviation
      qdiff_hess = dqdiff.transpose()*dqdiff; 
      for (int i = 0; i < pin_model->nv; ++i) {
        for (int j = 0; j < pin_model->nv; ++j) {
          hessian_triplets.push_back(Triplet<double>(var_idx + i, var_idx + j, qdiff_hess(i, j)));
        }
      }

      if (step < steps - 1) {
        // Velocity equality constraint
        pin::difference(*pin_model, q_trj[step], q_trj[step + 1], v);
        pin::dDifference(*pin_model, q_trj[step], q_trj[step + 1], dv.leftCols(pin_model->nv), pin::ArgumentPosition::ARG0);
        pin::dDifference(*pin_model, q_trj[step], q_trj[step + 1], dv.rightCols(pin_model->nv), pin::ArgumentPosition::ARG1);

        for (int i = 0; i < pin_model->nv; ++i) {
          for (int j = 0; j < pin_model->nv; ++j) {
            linearMatrix_triplets.push_back(Triplet<double>(cidx, var_idx + i, dv(i, j)));
            int vars_in_this_step = vars_per_step;
            if (step == steps - 2) {
              vars_in_this_step -= pin_model->nv;
            }
            linearMatrix_triplets.push_back(Triplet<double>(cidx, var_idx + vars_in_this_step + i, dv(i, pin_model->nv + j)));
          }
          linearMatrix_triplets.push_back(Triplet<double>(cidx, var_idx + pin_model->nv + i, -1));
          ++cidx;
        }

        // Velocity deviation penalty

        // Gradient
        gradient.segment(var_idx + pin_model->nv, pin_model->nv) += v_cost*(v - v_ref[step]);

        // Hessian
        for (int i = 0; i < pin_model->nv; ++i) {
          hessian_triplets.push_back(Triplet<double>(var_idx + pin_model->nv + i, var_idx + pin_model->nv + i, v_cost));
        }
      }

      if (step < steps - 2) {
        // Acceleration equality constraint
        vdot = (v_next - v)/dt;

        for (int i = 0; i < pin_model->nv; ++i) {
          linearMatrix_triplets.push_back(Triplet<double>(cidx, var_idx + pin_model->nv + i, -1/dt));
          linearMatrix_triplets.push_back(Triplet<double>(cidx, var_idx + vars_per_step + pin_model->nv + i, 1/dt));
          linearMatrix_triplets.push_back(Triplet<double>(cidx, var_idx + 2*pin_model->nv + i, -1));
          ++cidx;
        }

        // Acceleration deviation penalty

        // Gradient
        gradient.segment(var_idx + 2*pin_model->nv, pin_model->nv) += vdot_cost*(vdot - vdot_ref[step]);

        // Hessian
        for (int i = 0; i < pin_model->nv; ++i) {
          hessian_triplets.push_back(Triplet<double>(var_idx + 2*pin_model->nv + i, var_idx + 2*pin_model->nv + i, vdot_cost));
        }
      }

      v_next = v;
      dv_next = dv;

      // Penetration constraints
      if (num_contact_pairs) {
        pin::forwardKinematics(*pin_model, pin_data, q_trj[step]);
        pin::computeJointJacobians(*pin_model, pin_data);
        for (int j = 0; j < pin_model->njoints; ++j) {
          pin::getJointJacobian(*pin_model, pin_data, j, pin::ReferenceFrame::LOCAL_WORLD_ALIGNED, joint_jacobians[j]);
        }
        pin::computeDistances(*pin_model, pin_data, *pin_geom, geom_data);

        for (int i = 0; i < pin_geom->collisionPairs.size(); ++i) {
          distances(i) = geom_data.distanceResults[i].min_distance;
          lowerBound(cidx + i) = -distances(i);
          upperBound(cidx + i) = 10000;
          if (geom_data.distanceResults[i].min_distance < 0) {
            normal = geom_data.distanceResults[i].normal;
          } else {
            normal = geom_data.distanceResults[i].nearest_points[1] - geom_data.distanceResults[i].nearest_points[0];
          }
          normal.normalize();

          int jidx0 = pin_geom->geometryObjects[pin_geom->collisionPairs[i].first].parentJoint;
          t0 = geom_data.distanceResults[i].nearest_points[0] - pin_data.oMi[jidx0].translation();
          adj0.rightCols<3>() = -pin::skew(t0);

          int jidx1 = pin_geom->geometryObjects[pin_geom->collisionPairs[i].second].parentJoint;
          t1 = geom_data.distanceResults[i].nearest_points[1] - pin_data.oMi[jidx1].translation();
          adj1.rightCols<3>() = -pin::skew(t1);

          distance_jacobian.row(i) = normal.transpose()*(adj1*joint_jacobians[jidx1] - adj0*joint_jacobians[jidx0]);
        }

        for (int i = 0; i < num_contact_pairs; ++i) {
          for (int j = 0; j < pin_model->nv; ++j) {
            linearMatrix_triplets.push_back(Triplet<double>(cidx, var_idx + j, distance_jacobian(i, j)));
          }
          ++cidx;
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

    for (int step = 0; step < steps; ++step) {
      var_idx = step*vars_per_step;
      if (step == steps - 1) {
        var_idx -= pin_model->nv;
      }
      pin::integrate(*pin_model, q_trj[step], primal.segment(var_idx, pin_model->nv), q_trj[step]);
    }
  }
  return true;
}
