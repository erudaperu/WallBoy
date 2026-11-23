// Build with -I pointing to your export directory. No PyTorch/ONNX libraries.
#include "policy.hpp"
#include <cmath>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: verify parity.tsv\n";
        return 2;
    }
    std::ifstream input(argv[1]);
    if (!input) return 2;
    int count = 0;
    float max_error = 0.0f;
    while (true) {
        wallboy_ai::Observation observation{};
        if (!(input >> observation[0])) break;
        for (int i = 1; i < 7; ++i) {
            if (!(input >> observation[i])) return 2;
        }
        float q0, q1;
        int expected;
        if (!(input >> q0 >> q1 >> expected)) return 2;
        const auto actual = wallboy_ai::q_values(observation);
        const float error = std::max(std::abs(actual[0]-q0), std::abs(actual[1]-q1));
        if (!std::isfinite(error) || error > 0.001f || wallboy_ai::action(observation) != expected) {
            std::cerr << "Parity failure at sample " << count << ", error " << error << '\n';
            return 1;
        }
        max_error = std::max(max_error, error);
        ++count;
    }
    if (count == 0) return 2;
    std::cout << "PASS: " << count << " observations; all actions match; maximum Q error "
              << max_error << '\n';
}
