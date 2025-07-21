#include "ZeroOrderOptimize.hpp"
#include <random>

namespace mllm {
namespace optim {

vector<Tensor *> ZeroOrderOptimizer::weights_to_optimize;

vector<vector<float>> ZeroOrderOptimizer::delta_vec;

}
} // namespace mllm::optim