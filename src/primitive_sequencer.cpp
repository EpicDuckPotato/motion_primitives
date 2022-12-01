#include "primitive_sequencer.h"

PrimitiveSequencer::PrimitiveSequencer(const PrimitiveLibrary &library, boost::shared_ptr<pin::Model> pin_model, boost::shared_ptr<pin::GeometryModel> pin_geom) : library(library), pin_model(pin_model), pin_geom(pin_geom) {
  pin_data = pin::Data(*pin_model);
  geom_data = pin::GeometryData(*pin_geom);
}

void PrimitiveSequencer::sequence(vector<VectorXd> &solution, const R3Path &r3_path, const Ref<const VectorXd> &q0) {
  Vector3d delta_pos = Vector3d::Zero();
  boost::shared_ptr<PrimitiveTreeNode> root_node = boost::make_shared<PrimitiveTreeNode>(0, q0, library, r3_path, 1, delta_pos, 0, -1, -1, pin_model, pin_geom, pin_data, geom_data);
  vector<boost::shared_ptr<Node>> solution_nodes;
  if (!tree_search(solution_nodes, root_node)) {
    std::cout << "No solution found" << std::endl;
  }

  // Get total solution
  solution.clear();
  solution.push_back(q0);
  for (int s = 0; s < solution_nodes.size() - 1; ++s) {
    boost::shared_ptr<PrimitiveTreeNode> node = boost::static_pointer_cast<PrimitiveTreeNode>(solution_nodes[s]);
    boost::shared_ptr<PrimitiveTreeNode> next_node = boost::static_pointer_cast<PrimitiveTreeNode>(solution_nodes[s + 1]);
    int i = next_node->get_generating_primitive();
    const VectorXd &q = node->get_q();
    const Vector3d &delta_pos = next_node->get_delta_pos();

    Vector3d head_direction = Quaterniond(q.segment<4>(3)).toRotationMatrix().col(2);
    head_direction(2) = 0;

    // Get this node's primitive
    Vector3d old_head_direction = Quaterniond(library[i][0].segment<4>(3)).toRotationMatrix().col(2);
    old_head_direction(2) = 0;
    Quaterniond quat_transform = Quaterniond::FromTwoVectors(old_head_direction, head_direction);
    for (int j = 1; j < library[i].size(); ++j) {
      VectorXd q_primitive = library[i][j];
      q_primitive.head<3>() = q.head<3>() + quat_transform*(library[i][j].head<3>() - library[i][0].head<3>()) + j*delta_pos/(library[i].size() - 1);
      q_primitive.segment<4>(3) = (quat_transform*Quaterniond(q_primitive.segment<4>(3))).coeffs();
      solution.push_back(q_primitive);
    }
  }
}
