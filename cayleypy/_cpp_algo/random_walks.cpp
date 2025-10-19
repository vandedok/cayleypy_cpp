#include "random_walks.h"

#include <iostream>
#include <vector>
#include <random>
#include <torch/extension.h>

namespace py = pybind11;

using std::vector;
using std::cout;
using torch::Tensor;
using torch::tensor;

const int* get_row_ptr(const vector<int>& mat, int row, int row_size) {
    return &mat[static_cast<size_t>(row) * row_size];
}

void apply_gen(
    const vector<int>& state, 
    const int gen_k, 
    const vector<int>& gens, 
    const int state_size,
    vector<int>& result
) {
    const int* gen_ptr = get_row_ptr(gens, gen_k, state_size);
    for(int i=0; i<state_size; i++){
        result[i] = state[gen_ptr[i]];
    }   
} 


struct Walks {
    int num_walks;
    int walk_length;
    int state_size;
    vector<int> states;
    vector<int> gen_choices;
    vector<int> distances;

    Walks(int num_walks, int walks_len, int state_size)
        : num_walks(num_walks),
          walk_length(walks_len),
          state_size(state_size)
    {
        states.reserve(num_walks * walk_length * state_size);
        gen_choices.reserve(num_walks * walk_length);
        distances.reserve(num_walks * walk_length);
    }

    const int* get_state_ptr(int walk, int step) const {
        size_t offset = static_cast<size_t>((walk * walk_length + step) * state_size);
        return &states[offset];
    }

    vector<int> get_state_copy(int walk_k, int step_k) const {
        const int* ptr = get_state_ptr(walk_k, step_k);
        return vector<int>(ptr, ptr + state_size);
    }

    int get_choice(int walk, int step) const {
        return gen_choices[walk * walk_length + step];
    }
};

Walks generate_random_walk(const vector<int> gens, const vector<int> central_state, const int num_walks, const int walks_len, const int num_gens, const int state_size){
    
    int n_choices{num_walks*walks_len};  // how many numbers
    int low{0}; 
    int high{num_gens-1};  

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(low, high);
    Walks rw_result(num_walks, walks_len, state_size); 

    vector<int> current_state(central_state);
    vector<int> next_state(state_size);
    int choice{0};
    int walks_index_offset{0};

    for (int i = 0; i < n_choices; i++) {
        rw_result.gen_choices.push_back(dist(gen));
    }

    for(int iw = 0; iw < num_walks; iw++) {
        current_state = central_state;
        walks_index_offset = iw * walks_len;
        for(int js = 0; js < walks_len; js++){
            apply_gen(current_state, rw_result.gen_choices[walks_index_offset + js], gens, state_size, next_state);
            current_state.swap(next_state);
            for (int k = 0; k < state_size; k++)
                rw_result.states.push_back(current_state[k]);
        }
    }

    return rw_result;
}


Tensor _random_walks_classic_cpp(const vector<int> gens, const vector<int> central_state, const int num_gens, const int state_size, const int num_walks, const int walks_len) {
    // print_vec(gens);

    int n_choices{num_walks*walks_len};  
    int low{0}; 
    int high{num_gens-1};  

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(low, high);
    Walks rw_result(num_walks, walks_len, state_size); 

    vector<int> current_state(central_state);
    vector<int> next_state(state_size);
    int choice{0};


    for (int i = 0; i < n_choices; i++) {
        rw_result.gen_choices.push_back(dist(gen));
    }

    for(int iw = 0; iw < num_walks; iw++) {
        current_state = central_state;
        for(int js = 0; js < walks_len; js++){
            apply_gen(current_state, rw_result.gen_choices[iw*walks_len + js], gens, state_size, next_state);
            current_state.swap(next_state);
            for (int k = 0; k < state_size; k++) {
                rw_result.states.push_back(current_state[k]);
                rw_result.distances.push_back(k);
            }
        }
    }
    
    auto states_tensor = tensor(rw_result.states, torch::dtype(torch::kInt32));
    // return rw_result by reference? -- less safe, might be more efficient
    return states_tensor; 
}
