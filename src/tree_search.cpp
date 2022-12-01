#include "tree_search.h"
#include <vector>
#include <queue>
#include <iostream>

bool tree_search(vector<boost::shared_ptr<Node>> &solution, const boost::shared_ptr<Node> &root) {
  priority_queue<boost::shared_ptr<Node>, vector<boost::shared_ptr<Node>>, compare_nodes> open_list;   
  vector<boost::shared_ptr<Node>> nodes;
  nodes.push_back(root);
  open_list.push(root);
  int max_iter = 1000;
  int print_interval = max_iter/10;
  boost::shared_ptr<Node> solution_node;
  bool solved = false;
  for (int i = 0; i < max_iter; ++i) {
    if (i%print_interval == 0) {
      std::cout << "Starting tree search iteration " << i << std::endl;
    }
    boost::shared_ptr<Node> node = open_list.top();
    open_list.pop();
    if (node->is_solution_candidate() && node->evaluate_solution(nodes)) {
      solved = true;
      if (solution_node == nullptr || solution_node->get_cost_lb() > node->get_cost_lb()) {
        solution_node = node;
      }
      if (open_list.top()->get_cost_lb() >= node->get_cost_lb()) {
        break;
      }
    }
    vector<boost::shared_ptr<Node>> successors;
    node->get_successors(successors, nodes.size());
    for (auto successor : successors) {
      nodes.push_back(successor);
      open_list.push(successor);
    }

    if (!solved && (open_list.size() == 0 || i == max_iter - 1)) {
      // Return solution corresponding to the most recent expansion, for debugging purposes
      solution_node = nodes.back();
      break;
    }
  }

  // Populate solution
  solution.clear();
  while (true) {
    solution.push_back(solution_node);
    int id = solution_node->get_parent_id();
    if (id == -1) {
      break;
    }
    solution_node = nodes[id];
  }
  std::reverse(solution.begin(), solution.end());
  return solved;
}
