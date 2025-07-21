#pragma once

#include "Tensor.hpp"
#include <random>
#include <string>

namespace mllm {
namespace optim {

enum class PERTUR_TYPE {
    ADD,
    SUB
};

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(float loss = 0) {};                 // Perform a single optimization step
    virtual void zero_grad() {};                          // Reset gradients for the parameters
    virtual void load_state(const std::string &state) {}; // Load optimizer
    virtual void save_state(const std::string &state) {}; // Save optimizer state
};

class ZeroOrderOptimizer : public Optimizer {
    static vector<Tensor *> weights_to_optimize;
    static vector<vector<float>> delta_vec; // Perturbation vector
    int vector_idx = 0;                     // Index for the current weight in weights_to_optimize

    float loss;

public:
    virtual ~ZeroOrderOptimizer() = default;

    static void registerWeight(Tensor &weight) {
        weights_to_optimize.push_back(&weight);
        std::cout << "Registered weight: " << weight.name() << std::endl;
        std::cout << "Weight shape: " << weight.dimension() << std::endl;
        delta_vec.emplace_back(std::vector<float>(weight.dimension(), 0.0f));
    }

    void initRandomVector() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        for (auto &vec : delta_vec) {
            for (auto &val : vec) {
                val = dis(gen);
            }
        }
    }

    void applyPerturbation(PERTUR_TYPE type) {
        for (int i = 0; i < weights_to_optimize.size(); ++i) {
            auto &weight = *weights_to_optimize[i];
            auto &perturbation = delta_vec[i];

            for (int dim_idx = 0; dim_idx < weight.dimension(); ++dim_idx) {
                if (type == PERTUR_TYPE::ADD) {
                    weight.setDataAt<float>(0, 0, vector_idx, dim_idx, weight.d<float>(0, vector_idx, 0, dim_idx) + perturbation[dim_idx]);
                } else if (type == PERTUR_TYPE::SUB) {
                    weight.setDataAt<float>(0, 0, vector_idx, dim_idx, weight.d<float>(0, vector_idx, 0, dim_idx) - perturbation[dim_idx]);
                }
            }
        }
    }

    void removePerturbation(PERTUR_TYPE type) {
        for (int i = 0; i < weights_to_optimize.size(); ++i) {
            auto &weight = *weights_to_optimize[i];
            auto &perturbation = delta_vec[i];

            for (int dim_idx = 0; dim_idx < weight.dimension(); ++dim_idx) {
                if (type == PERTUR_TYPE::ADD) {
                    weight.setDataAt<float>(0, 0, vector_idx, dim_idx, weight.d<float>(0, vector_idx, 0, dim_idx) - perturbation[dim_idx]);
                } else if (type == PERTUR_TYPE::SUB) {
                    weight.setDataAt<float>(0, 0, vector_idx, dim_idx, weight.d<float>(0, vector_idx, 0, dim_idx) + perturbation[dim_idx]);
                }
            }
        }
    }

    std::vector<std::vector<float>> log_softmax(Tensor &logits) {
        int seq_len = logits.sequence();
        int vocab_size = logits.dimension();

        std::vector<std::vector<float>> log_probs(seq_len, std::vector<float>(vocab_size));

        for (int i = 0; i < seq_len; ++i) {
            float max_logit = -INFINITY;
            for (int j = 0; j < vocab_size; ++j) {
                max_logit = std::max(max_logit, logits.d<float>(0, i, 0, j));
            }
            float sum_exp = 0.0;
            for (int j = 0; j < vocab_size; ++j) {
                sum_exp += std::exp(logits.d<float>(0, i, 0, j) - max_logit);
            }
            for (int j = 0; j < vocab_size; ++j) {
                log_probs[i][j] = logits.d<float>(0, i, 0, j) - max_logit - std::log(sum_exp);
            }
        }
        return log_probs;
    }

    double compute_nll_loss(
        Tensor &logits,                           // [seq_len][vocab_size]
        const std::vector<int> &rewriting_targets // [seq_len], -100 = masked
    ) {
        int seq_len = logits.sequence();
        int vocab_size = logits.dimension();

        if (rewriting_targets.size() != seq_len) {
            throw std::invalid_argument("rewriting_targets size must match logits sequence length");
        }

        auto log_probs = log_softmax(logits);

        double loss_sum = 0.0;
        int valid_token_count = 0;

        for (int i = 0; i < seq_len; ++i) {
            int target_idx = rewriting_targets[i];
            if (target_idx != -100) {
                float log_prob = log_probs[i][target_idx];

                std::cout << "log prob for token " << i << " (target " << target_idx << "): " << log_prob << std::endl;

                loss_sum -= log_prob;
                valid_token_count += 1;
            }
        }

        if (valid_token_count == 0) return 0.0f;
        return loss_sum / valid_token_count;
    }

    void setVectorIdx(int idx) {
        vector_idx = idx;
    }

    // 计算向量的 L2 范数
    float compute_norm(const std::vector<float> &vec) {
        float norm = 0.0f;
        for (float val : vec) {
            norm += val * val;
        }
        return std::sqrt(norm);
    }

    // 约束向量 L2 范数
    void clip_norm(std::vector<float> &vec, float max_norm) {
        float norm = compute_norm(vec);
        if (norm > max_norm && norm > 1e-6f) {
            float scale = max_norm / norm;
            for (float &val : vec) {
                val *= scale;
            }
        }
    }

    // 更新某一个 delta_vec[i]
    void mobiedit_zero_order_optimization(
        float loss_plus,  // 正向loss
        float loss_minus, // 反向loss
        float zo_eps = 1e-3,
        float learning_rate = 0.05,
        float max_norm = 1.0) {
        for (int delta_index = 0; delta_index < delta_vec.size(); ++delta_index) {
            std::vector<float> &delta = delta_vec[delta_index];
            auto dim = delta.size();
            std::vector<float> gradient_est(dim, 0.0f);

            float grad_coeff = (loss_plus - loss_minus) / (2 * zo_eps);

            std::cout << "Gradient coefficient: " << grad_coeff << std::endl;

            for (size_t j = 0; j < dim; ++j) {
                gradient_est[j] += grad_coeff * delta[j];
            }

            // 更新 delta 向量
            for (size_t j = 0; j < dim; ++j) {
                delta[j] -= learning_rate * gradient_est[j];
            }

            // 范数裁剪
            clip_norm(delta, max_norm);
        }
    }
};

}
} // namespace mllm::optim