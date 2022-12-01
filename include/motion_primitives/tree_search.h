#pragma once
#include <boost/shared_ptr.hpp>

using namespace std;

class Node {
  public:
    // If parent_id == -1, this is the root of the tree
    Node(int id, int parent_id, double cost_lb) : id(id), parent_id(parent_id), cost_lb(cost_lb) {
    }

    // Gets lower bound on cost of a solution ending
    // with this node
    double get_cost_lb() {
      return cost_lb;
    }

    virtual bool is_solution_candidate() = 0;

    // If this is a solution, updates the cost
    // lower bound to the actual cost and returns true.
    // Otherwise, returns false without updating the lower
    // bound
    virtual bool evaluate_solution(const vector<boost::shared_ptr<Node>> &nodes) = 0;

    virtual void get_successors(vector<boost::shared_ptr<Node>> &successors, int start_id) = 0;

    int get_parent_id() const {
      return parent_id;
    }

  protected:
    int id;
    int parent_id;
    double cost_lb;
};

struct compare_nodes {
  bool operator() (boost::shared_ptr<Node> &node1,
                   boost::shared_ptr<Node> &node2) {
    return node1->get_cost_lb() > node2->get_cost_lb();
  }
};

bool tree_search(vector<boost::shared_ptr<Node>> &solution, const boost::shared_ptr<Node> &root);
