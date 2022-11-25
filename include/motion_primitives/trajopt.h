#pragma once
#include <pinocchio/fwd.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>
#include "pinocchio/algorithm/geometry.hpp"

using namespace Eigen;
using namespace std;
namespace pin = pinocchio;

// Tries to project the robot's configuration to the given constraint
class Trajopt {
  public:
    Trajopt(boost::shared_ptr<pin::Model> pin_model, boost::shared_ptr<pin::GeometryModel> pin_geom);
    bool optimize(std::vector<VectorXd>& q_trj, const std::vector<VectorXd>& q_ref, const Ref<const VectorXd> &qstart, const Ref<const Vector3d> &goal_pos, double dt, double q_cost, double v_cost, double vdot_cost, int max_outer);
  private:
    boost::shared_ptr<pin::Model> pin_model;
    boost::shared_ptr<pin::GeometryModel> pin_geom;
    boost::shared_ptr<OsqpEigen::Solver> solver;

    pin::Data pin_data;
    pin::GeometryData geom_data;

    VectorXd distances;
    MatrixXd distance_jacobian;

    vector<MatrixXd> joint_jacobians;
};
