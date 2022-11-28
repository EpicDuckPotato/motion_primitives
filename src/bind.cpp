#include "trajopt.h"
#include "primitive_sequencer.h"
#include <boost/python.hpp>
#include "eigenpy/eigenpy.hpp"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace bp = boost::python;

BOOST_PYTHON_MODULE(motion_primitives_bindings) {
  eigenpy::enableEigenPy();

  bp::class_<Primitive>("Primitive")
      .def(bp::vector_indexing_suite<Primitive>());

  bp::class_<PrimitiveLibrary>("PrimitiveLibrary")
      .def(bp::vector_indexing_suite<PrimitiveLibrary>());

  bp::class_<R3Path>("R3Path")
        .def(bp::vector_indexing_suite<R3Path>());

  bp::register_ptr_to_python<boost::shared_ptr<Trajopt>>();
  bp::class_<Trajopt>("Trajopt",
      bp::init<boost::shared_ptr<pin::Model>, boost::shared_ptr<pin::GeometryModel>>())
      .def("optimize", &Trajopt::optimize)
  ;

  bp::register_ptr_to_python<boost::shared_ptr<PrimitiveSequencer>>();
  bp::class_<PrimitiveSequencer>("PrimitiveSequencer",
      bp::init<const PrimitiveLibrary&>())
      .def("sequence", &PrimitiveSequencer::sequence)
  ;
}
