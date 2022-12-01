#pragma once

#include "primitive_tree_node.h"
#include "tree_search.h"

class PrimitiveSequencer {
  public:
    PrimitiveSequencer(const PrimitiveLibrary &library, boost::shared_ptr<pin::Model> pin_model, boost::shared_ptr<pin::GeometryModel> pin_geom);

    void sequence(vector<VectorXd> &solution, const R3Path &r3_path, const Ref<const VectorXd> &q0);

  private:
    const PrimitiveLibrary &library;
    boost::shared_ptr<pin::Model> pin_model;
    boost::shared_ptr<pin::GeometryModel> pin_geom;
    pin::Data pin_data;
    pin::GeometryData geom_data;
};
