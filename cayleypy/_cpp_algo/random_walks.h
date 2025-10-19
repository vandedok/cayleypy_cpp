#pragma once

#include <vector>
#include <torch/extension.h>

// Only declare what you need to expose to the binding.
// Keep std/torch types fully qualified in headers.
torch::Tensor _random_walks_classic_cpp(
    const std::vector<int> gens,
    const std::vector<int> central_state,
    const int num_gens,
    const int state_size,
    const int num_walks,
    const int walks_len
);