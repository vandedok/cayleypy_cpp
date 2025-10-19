#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "random_walks.h"
#include "random_walks_torch.h"

namespace py = pybind11;

// Create the Python module (name must match the extension module filename)
PYBIND11_MODULE(_cpp_algo, m) {
    m.doc() = "pybind11 bindings for CayleyPy C++ algorithms";
    m.def("_random_walks_classic_cpp",
          &_random_walks_classic_cpp,
          "Vanilla C++ random walks function");

    m.def("_random_walks_torch_cpp",
          &_random_walks_torch_cpp,
          "Torch Tensor-based C++ random walks function");
}
