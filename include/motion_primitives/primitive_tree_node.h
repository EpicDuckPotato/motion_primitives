#pragma once

#include "tree_search.h"
#include <Eigen/Dense>
#include <boost/make_shared.hpp>

using namespace Eigen;

typedef vector<Vector3d> R3Path;
typedef vector<VectorXd> Primitive;
typedef vector<Primitive> PrimitiveLibrary;

// TODO: incorporate penetration depth into the cost
class PrimitiveTreeNode : public Node {
  public:
    PrimitiveTreeNode(double cost, const VectorXd &q, const PrimitiveLibrary &library, const R3Path &r3_path, int path_start, const Vector3d &delta_pos, int id, int parent_id, int generating_primitive) : Node(id, parent_id, cost), q(q), library(library), r3_path(r3_path), path_start(path_start), delta_pos(delta_pos), generating_primitive(generating_primitive) {
      reached_goal = (q.head<3>() - r3_path.back()).norm() < 0.1;
    }

    virtual bool is_solution_candidate() override {
      return reached_goal;
    }

    virtual bool evaluate_solution(const vector<boost::shared_ptr<Node>> &nodes) override {
      return reached_goal;
    }

    virtual void get_successors(vector<boost::shared_ptr<Node>> &successors, int start_id) {
      successors.clear();
      for (int i = 0; i < library.size(); ++i) {
        // Transform the primitive's final head position,
        // find the closest point on the path,
        // make that the intermediate goal that we warp to
        Quaterniond quat = Quaterniond(q.segment<4>(3))*Quaterniond(library[i][0].segment<4>(3)).conjugate();
        double min_dist = 100;
        int min_path_step = 0;
        Vector3d primitive_end = q.head<3>() + quat*library[i].back().head<3>();
        Vector3d min_delta_pos;
        for (int path_step = path_start; path_step < r3_path.size(); ++path_step) {
          Vector3d cand_delta_pos = r3_path[path_step] - primitive_end;
          double dist = cand_delta_pos.norm();
          if (dist < min_dist) {
            min_dist = dist;
            min_path_step = path_step;
            min_delta_pos = cand_delta_pos;
          }
        }

        VectorXd qnext = library[i].back();
        qnext.head<3>() = primitive_end + min_delta_pos;
        qnext.segment<4>(3) = (quat*Quaterniond(library[i].back().segment<4>(3))).coeffs();

        double cost = 0;
        for (int j = 0; j < library[i].size(); ++j) {
          cost += (min_dist*j)/(library[i].size());
        }

        successors.push_back(boost::make_shared<PrimitiveTreeNode>(min_dist, qnext, library, r3_path, min_path_step, min_delta_pos, start_id + i, id, i));
      }
    }

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
};
