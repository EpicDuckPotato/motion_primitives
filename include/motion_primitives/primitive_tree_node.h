#pragma once

#include <pinocchio/fwd.hpp>
#include <pinocchio/container/boost-container-limits.hpp>
#include "pinocchio/algorithm/geometry.hpp"
#include "tree_search.h"
#include <Eigen/Dense>
#include <boost/make_shared.hpp>

using namespace Eigen;
namespace pin = pinocchio;

typedef vector<Vector3d> R3Path;
typedef vector<VectorXd> Primitive;
typedef vector<Primitive> PrimitiveLibrary;

// TODO: incorporate penetration depth into the cost
class PrimitiveTreeNode : public Node {
  public:
    PrimitiveTreeNode(double cost, const VectorXd &q, const PrimitiveLibrary &library, const R3Path &r3_path, int path_start, const Vector3d &delta_pos, int id, int parent_id, int generating_primitive, boost::shared_ptr<pin::Model> pin_model, boost::shared_ptr<pin::GeometryModel> pin_geom, pin::Data &pin_data, pin::GeometryData &geom_data);

    virtual bool is_solution_candidate() override;

    virtual bool evaluate_solution(const vector<boost::shared_ptr<Node>> &nodes) override;

    virtual void get_successors(vector<boost::shared_ptr<Node>> &successors, int start_id);

    // Stuff that'll be useful once we find a solution
    const Vector3d &get_delta_pos() const {
      return delta_pos;
    }

    int get_generating_primitive() const {
      return generating_primitive;
    }

    const VectorXd &get_q() const {
      return q;
    }

  protected:
    const PrimitiveLibrary &library;
    VectorXd q;
    bool reached_goal;
    const R3Path &r3_path;
    int path_start;

    // Stuff that'll be useful once we find a solution
    Vector3d delta_pos;
    int generating_primitive;

    boost::shared_ptr<pin::Model> pin_model;
    boost::shared_ptr<pin::GeometryModel> pin_geom;
    pin::Data &pin_data;
    pin::GeometryData &geom_data;
};
