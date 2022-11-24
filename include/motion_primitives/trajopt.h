#pragma once
#include <pinocchio/fwd.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>
#include "itp_contact/dcol_wrapper.h"

using namespace Eigen;
using namespace std;
namespace pin = pinocchio;

// Tries to project the robot's configuration to the given constraint
class Trajopt {
  public:
    Trajopt(boost::shared_ptr<pin::Model> pin_model, boost::shared_ptr<ContactInfo> contact_info);
    bool optimize(std::vector<VectorXd>& q_trj, const std::vector<VectorXd>& q_ref, const Ref<const VectorXd> &qstart, const Ref<const Vector3d> &goal_pos, double dt, double q_cost, double v_cost, double vdot_cost);
  private:
    boost::shared_ptr<pin::Model> pin_model;
    boost::shared_ptr<ContactInfo> contact_info;
    boost::shared_ptr<OsqpEigen::Solver> solver;

    DCOLWrapper dcol_wrapper;
    VectorXd distances;
    MatrixXd distance_jacobian;
};
