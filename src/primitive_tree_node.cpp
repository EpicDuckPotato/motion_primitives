#include "primitive_tree_node.h"

PrimitiveTreeNode::PrimitiveTreeNode(double cost, const VectorXd &q, const PrimitiveLibrary &library, const R3Path &r3_path, int path_start, const Vector3d &delta_pos, int id, int parent_id, int generating_primitive, boost::shared_ptr<pin::Model> pin_model, boost::shared_ptr<pin::GeometryModel> pin_geom, pin::Data &pin_data, pin::GeometryData &geom_data) : Node(id, parent_id, cost), q(q), library(library), r3_path(r3_path), path_start(path_start), delta_pos(delta_pos), generating_primitive(generating_primitive), pin_model(pin_model), pin_geom(pin_geom), pin_data(pin_data), geom_data(geom_data) {
  reached_goal = (q.head<3>() - r3_path.back()).norm() < 0.1;
}

bool PrimitiveTreeNode::is_solution_candidate() {
  return reached_goal;
}

bool PrimitiveTreeNode::evaluate_solution(const vector<boost::shared_ptr<Node>> &nodes) {
  return reached_goal;
}

void PrimitiveTreeNode::get_successors(vector<boost::shared_ptr<Node>> &successors, int start_id) {
  successors.clear();
  Vector3d head_direction = Quaterniond(q.segment<4>(3)).toRotationMatrix().col(2);
  head_direction(2) = 0;
  for (int i = 0; i < library.size(); ++i) {
    // Transform the primitive's final head position accounting for symmetry wrt yaw and translation,
    // find the closest point on the path,
    // make that the intermediate goal that we warp to
    double min_dist = 100;
    int min_path_step = 0;
    // Where would the primitive end if we didn't warp it?
    Vector3d old_head_direction = Quaterniond(library[i][0].segment<4>(3)).toRotationMatrix().col(2);
    old_head_direction(2) = 0;
    Quaterniond quat_transform = Quaterniond::FromTwoVectors(old_head_direction, head_direction);
    Vector3d primitive_end_nominal = q.head<3>() + quat_transform*(library[i].back().head<3>() - library[i][0].head<3>());
    Vector3d min_delta_pos;
    for (int path_step = path_start + 1; path_step < r3_path.size(); ++path_step) {
      // How much we'd have to warp the final position of the primitive to connect it to the path at this step
      Vector3d cand_delta_pos = r3_path[path_step] - primitive_end_nominal;
      double dist = cand_delta_pos.norm();
      if (dist < min_dist) {
        min_dist = dist;
        min_path_step = path_step;
        min_delta_pos = cand_delta_pos;
      }
    }

    VectorXd qnext = library[i].back();
    qnext.head<3>() = primitive_end_nominal + min_delta_pos;
    qnext.segment<4>(3) = (quat_transform*Quaterniond(library[i].back().segment<4>(3))).coeffs();

    double worst_pen = 0;
    for (int j = 0; j < library[i].size(); ++j) {
      VectorXd q_primitive = library[i][j];
      q_primitive.head<3>() = q.head<3>() + quat_transform*(library[i][j].head<3>() - library[i][0].head<3>()) + j*min_delta_pos/(library[i].size() - 1);
      q_primitive.segment<4>(3) = (quat_transform*Quaterniond(library[i][j].segment<4>(3))).coeffs();
      pin::computeDistances(*pin_model, pin_data, *pin_geom, geom_data, q_primitive);
      for (auto res : geom_data.distanceResults) {
        if (res.min_distance < worst_pen) {
          worst_pen = res.min_distance;
        }
      }
    }
    double successor_cost = cost_lb + min_dist - worst_pen;

    successors.push_back(boost::make_shared<PrimitiveTreeNode>(successor_cost, qnext, library, r3_path, min_path_step, min_delta_pos, start_id + i, id, i, pin_model, pin_geom, pin_data, geom_data));
  }
}
