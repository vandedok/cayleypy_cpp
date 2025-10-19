#include "random_walks_torch.h"

#include <torch/extension.h>
#include <cstdint>
#include <utility>   // std::swap

using torch::Tensor;

Tensor _random_walks_torch_cpp(
    Tensor gens,            
    Tensor central_state,  
    const int64_t num_walks,
    const int64_t walks_len)   
{
    TORCH_CHECK(gens.device().is_cpu(), "CPU only implementation (for now).");
    TORCH_CHECK(central_state.device().is_cpu(), "CPU only implementation (for now).");
    TORCH_CHECK(gens.dtype() == torch::kLong, "gens must be int64 (LongTensor).");
    TORCH_CHECK(central_state.dtype() == torch::kLong, "central_state must be int64 (LongTensor).");
    TORCH_CHECK(gens.dim() == 2, "gens must be 2D [num_gens, state_size].");
    TORCH_CHECK(central_state.dim() == 1, "central_state must be 1D [state_size].");
    TORCH_CHECK(num_walks >= 0 && walks_len >= 0, "num_walks and walks_len must be non-negative.");

    const int64_t num_gens   = gens.size(0);
    const int64_t state_size = gens.size(1);

    gens = gens.contiguous();
    central_state = central_state.contiguous();

    auto tensor_options = central_state.options();
    Tensor states = torch::empty({num_walks, walks_len, state_size}, tensor_options);
    Tensor choices = torch::randint(num_gens, {num_walks, walks_len}, tensor_options.dtype(torch::kLong));

    Tensor current = torch::empty_like(central_state);  
    Tensor next = torch::empty_like(central_state);  

    const int64_t* gens_ptr = gens.data_ptr<int64_t>();
    const int64_t* central_state_ptr = central_state.data_ptr<int64_t>();
    const int64_t* choices_ptr = choices.data_ptr<int64_t>();
    int64_t* states_ptr = states.data_ptr<int64_t>();
    int64_t* current_ptr = current.data_ptr<int64_t>();
    int64_t* next_ptr = next.data_ptr<int64_t>();


    for (int64_t iw = 0; iw < num_walks; ++iw) {
        std::memcpy(current_ptr, central_state_ptr, state_size * sizeof(int64_t));
        const int64_t offset = iw * walks_len;

        for (int64_t js = 0; js < walks_len; ++js) {
            const int64_t choice = choices_ptr[offset + js];
            const int64_t* gen_row = gens_ptr + choice * state_size;

            int64_t* out_ptr = states_ptr + ((offset + js) * state_size);
            for (int64_t k = 0; k < state_size; ++k) {
                out_ptr[k] = current_ptr[gen_row[k]];
            }

            current_ptr = out_ptr;
        }
    }


    return states;
}