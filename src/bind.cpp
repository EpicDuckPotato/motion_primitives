#include "trajopt.h"
#include <boost/python.hpp>
#include "eigenpy/eigenpy.hpp"

namespace bp = boost::python;

BOOST_PYTHON_MODULE(motion_primitives_bindings) {
  eigenpy::enableEigenPy();

  bp::register_ptr_to_python<boost::shared_ptr<Trajopt>>();
  bp::class_<Trajopt>("Trajopt",
      bp::init<boost::shared_ptr<pin::Model>, boost::shared_ptr<ContactInfo>>())
      .def("optimize", &Trajopt::optimize)
  ;
}
