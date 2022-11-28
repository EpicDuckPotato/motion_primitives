#pragma once

#include "primitive_tree_node.h"
#include "tree_search.h"

class PrimitiveSequencer {
  public:
    PrimitiveSequencer(const PrimitiveLibrary &library) : library(library) {
    }

    void sequence(vector<VectorXd> &solution, const R3Path &r3_path, const Ref<const VectorXd> &q0) {
      Vector3d delta_pos = Vector3d::Zero();
      boost::shared_ptr<PrimitiveTreeNode> root_node = boost::make_shared<PrimitiveTreeNode>(0, q0, library, r3_path, 1, delta_pos, 0, -1, -1);
      vector<boost::shared_ptr<Node>> solution_nodes;
      tree_search(solution_nodes, root_node);
      if (solution_nodes.size() == 0) {
        std::cout << "No solution found" << std::endl;
        return;
      }

      // Get total solution
      solution.clear();
      solution.push_back(q0);
      for (int i = 0; i < solution_nodes.size() - 1; ++i) {
        boost::shared_ptr<PrimitiveTreeNode> node = boost::static_pointer_cast<PrimitiveTreeNode>(solution_nodes[i]);
        boost::shared_ptr<PrimitiveTreeNode> next_node = boost::static_pointer_cast<PrimitiveTreeNode>(solution_nodes[i + 1]);
        int primitive_idx = next_node->get_generating_primitive();
        const VectorXd &q = node->get_q();
        const Vector3d &delta_pos = next_node->get_delta_pos();

        // Get this node's primitive
        Quaterniond quat = Quaterniond(q.segment<4>(3))*Quaterniond(library[primitive_idx][0].segment<4>(3)).conjugate();
        for (int j = 1; j < library[primitive_idx].size(); ++j) {
          VectorXd q_primitive = library[primitive_idx][j];
          q_primitive.head<3>() = q.head<3>() + quat*q_primitive.head<3>() + j*delta_pos/(library[primitive_idx].size() - 1);
          q_primitive.segment<4>(3) = (quat*Quaterniond(q_primitive.segment<4>(3))).coeffs();
          solution.push_back(q_primitive);
        }
      }
    }

  private:
    const PrimitiveLibrary &library;
};
