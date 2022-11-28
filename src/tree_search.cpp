#include "tree_search.h"
#include <vector>
#include <queue>
#include <iostream>

void tree_search(vector<boost::shared_ptr<Node>> &solution, const boost::shared_ptr<Node> &root) {
  priority_queue<boost::shared_ptr<Node>, vector<boost::shared_ptr<Node>>, compare_nodes> open_list;   
  vector<boost::shared_ptr<Node>> nodes;
  nodes.push_back(root);
  open_list.push(root);
  int max_iter = 1000;
  for (int i = 0; i < max_iter; ++i) {
    if (open_list.size() == 0) {
      return;
    }
    boost::shared_ptr<Node> node = open_list.top();
    open_list.pop();
    if (node->is_solution_candidate() && 
        node->evaluate_solution(nodes) && 
        open_list.top()->get_cost_lb() >= node->get_cost_lb()) {

      solution.clear();
      while (true) {
        solution.push_back(node);
        int id = node->get_parent_id();
        if (id == -1) {
          break;
        }
        node = nodes[id];
      }
      std::reverse(solution.begin(), solution.end());
      return;
    }
    vector<boost::shared_ptr<Node>> successors;
    node->get_successors(successors, nodes.size());
    for (auto successor : successors) {
      nodes.push_back(successor);
      open_list.push(successor);
    }
  }
}
