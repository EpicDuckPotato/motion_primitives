#include "trajopt.h"

Trajopt::Trajopt(boost::shared_ptr<pin::Model> pin_model, boost::shared_ptr<ContactInfo> contact_info) : pin_model(pin_model), contact_info(contact_info), dcol_wrapper(pin_model, contact_info) {
  solver = boost::make_shared<OsqpEigen::Solver>();
  solver->settings()->setVerbosity(false);
  solver->settings()->setWarmStart(true);
  solver->settings()->setAdaptiveRhoInterval(50);

  distances = VectorXd::Zero(contact_info->get_num_contact_pairs());
  distance_jacobian = MatrixXd::Zero(contact_info->get_num_contact_pairs(), pin_model->nv);
}

bool Trajopt::optimize(std::vector<VectorXd>& q_trj, const std::vector<VectorXd>& q_ref, const Ref<const VectorXd> &qstart, const Ref<const Vector3d> &goal_pos, double dt, double q_cost, double v_cost, double vdot_cost) {
  int steps = q_ref.size();

  int num_contact_pairs = contact_info->get_num_contact_pairs();

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

  int num_decision_vars = pin_model->nv*steps;
  int num_constraints = 3 + pin_model->nv + contact_info->get_num_contact_pairs()*steps; // Goal constraint, start constraint, and non-penetration constraints

  VectorXd gradient(num_decision_vars);
  SparseMatrix<double> hessian(num_decision_vars, num_decision_vars);
  SparseMatrix<double> linearMatrix(num_constraints, num_decision_vars);
  vector<Triplet<double>> hessian_triplets;
  vector<Triplet<double>> linearMatrix_triplets;
  VectorXd lowerBound(num_constraints);
  VectorXd upperBound(num_constraints);

  Matrix3d rmat_last;
  VectorXd qdiff = VectorXd::Zero(pin_model->nv);
  MatrixXd dqdiff = MatrixXd::Zero(pin_model->nv, pin_model->nv);
  MatrixXd qdiff_hess = MatrixXd::Zero(pin_model->nv, pin_model->nv);

  VectorXd v = VectorXd::Zero(pin_model->nv);
  MatrixXd dv = MatrixXd::Zero(pin_model->nv, 2*pin_model->nv);
  MatrixXd v_hess = MatrixXd::Zero(2*pin_model->nv, 2*pin_model->nv);

  VectorXd v_next = VectorXd::Zero(pin_model->nv);
  MatrixXd dv_next = MatrixXd::Zero(pin_model->nv, 2*pin_model->nv);

  VectorXd vdot = VectorXd::Zero(pin_model->nv);
  MatrixXd dvdot = MatrixXd::Zero(pin_model->nv, 3*pin_model->nv);
  MatrixXd vdot_hess = MatrixXd::Zero(3*pin_model->nv, 3*pin_model->nv);

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
      pin::difference(*pin_model, q_ref[step], q_trj[step], qdiff);
      pin::dDifference(*pin_model, q_ref[step], q_trj[step], dqdiff, pin::ArgumentPosition::ARG1);

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
        pin::difference(*pin_model, q_trj[step], q_trj[step + 1], v);
        v /= dt;

        pin::dDifference(*pin_model, q_trj[step], q_trj[step + 1], dv.leftCols(pin_model->nv), pin::ArgumentPosition::ARG0);
        pin::dDifference(*pin_model, q_trj[step], q_trj[step + 1], dv.rightCols(pin_model->nv), pin::ArgumentPosition::ARG1);
        dv /= dt;

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
      int cidx = 3 + pin_model->nv + num_contact_pairs*step;
      dcol_wrapper.calcDiff(distances, distance_jacobian, q_trj[step]);
      lowerBound.segment(cidx, num_contact_pairs) = -distances;
      upperBound.segment(cidx, num_contact_pairs).setConstant(10000);
      for (int i = 0; i < num_contact_pairs; ++i) {
        for (int j = 0; j < pin_model->nv; ++j) {
          linearMatrix_triplets.push_back(Triplet<double>(cidx + i, startrow + j, distance_jacobian(i, j)));
        }
      }
    }

    hessian.setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
    linearMatrix.setFromTriplets(linearMatrix_triplets.begin(), linearMatrix_triplets.end());

    std::cout << MatrixXd(linearMatrix) << std::endl << std::endl;
    std::cout << lowerBound << std::endl << std::endl;
    std::cout << upperBound << std::endl << std::endl;

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
    }

    VectorXd primal = solver->getSolution();
    for (int step = 0; step < steps; ++step) {
      q_trj[step] = pin::integrate(*pin_model, q_trj[step], primal.segment(step*pin_model->nv, pin_model->nv));

      dcol_wrapper.calcDiff(distances, distance_jacobian, q_trj[step]);
      std::cout << distances.transpose() << std::endl;
    }

    // TODO: multiple iterations?
    return true;
  }
}
