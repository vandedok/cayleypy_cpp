#pragma once

#include <torch/extension.h>

using torch::Tensor;

Tensor _random_walks_torch_cpp(
    Tensor gens, 
    Tensor central_state, 
    const int64_t num_walks, 
    const int64_t walks_len
);