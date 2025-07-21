#include "Context.hpp"
#include "ZeroOrderOptimize.hpp"
#include "QNNBackend.hpp"
#include "Types.hpp"
#include "backends/cpu/CPUBackend.hpp"
#include "cmdline.h"
#include "models/qwen/configuration_qwen.hpp"
#include "models/qwen/modeling_qwen_npu_rome.hpp"
#include "models/qwen/modeling_qwen.hpp"
#include "models/qwen/tokenization_qwen.hpp"
#include "processor/PostProcess.hpp"
#include <nlohmann/json.hpp>

using namespace mllm;
using json = nlohmann::json;

class ROMEHyperParams {
public:
    // Method
    bool quantize;
    bool use_zo;
    bool use_random_prefix;
    bool use_eval;
    std::vector<int> layers;
    std::string fact_token;
    int v_num_grad_steps;
    float v_lr;
    int v_loss_layer;
    float v_weight_decay;
    float clamp_norm_factor;
    float kl_factor;
    bool mom2_adjustment;
    std::vector<std::vector<int>> context_template_length_params;

    // Module templates
    std::string rewrite_module_tmp;
    std::string layer_module_tmp;
    std::string mlp_module_tmp;
    std::string attn_module_tmp;
    std::string ln_f_module;
    std::string lm_head_module;

    // Statistics
    std::string mom2_dataset;
    int mom2_n_samples;
    std::string mom2_dtype;
    std::string alg_name;
    int device;
    std::string model_name;
    std::string stats_dir;

    // Optional fields with default values
    int max_length;
    bool model_parallel;
    bool fp16;

    // Constructor with default values
    ROMEHyperParams() :
        quantize(false), use_zo(false), use_random_prefix(false), use_eval(false),
        v_num_grad_steps(0), v_lr(0.0f), v_loss_layer(0), v_weight_decay(0.0f),
        clamp_norm_factor(0.0f), kl_factor(0.0f), mom2_adjustment(false),
        mom2_n_samples(0), device(0), max_length(40),
        model_parallel(false), fp16(false) {
    }

    static ROMEHyperParams from_json(const json &j) {
        ROMEHyperParams p;
        j.at("quantize").get_to(p.quantize);
        j.at("use_zo").get_to(p.use_zo);
        j.at("use_random_prefix").get_to(p.use_random_prefix);
        j.at("use_eval").get_to(p.use_eval);
        j.at("layers").get_to(p.layers);
        j.at("fact_token").get_to(p.fact_token);
        j.at("v_num_grad_steps").get_to(p.v_num_grad_steps);
        j.at("v_lr").get_to(p.v_lr);
        j.at("v_loss_layer").get_to(p.v_loss_layer);
        j.at("v_weight_decay").get_to(p.v_weight_decay);
        j.at("clamp_norm_factor").get_to(p.clamp_norm_factor);
        j.at("kl_factor").get_to(p.kl_factor);
        j.at("mom2_adjustment").get_to(p.mom2_adjustment);
        j.at("context_template_length_params").get_to(p.context_template_length_params);
        j.at("rewrite_module_tmp").get_to(p.rewrite_module_tmp);
        j.at("layer_module_tmp").get_to(p.layer_module_tmp);
        j.at("mlp_module_tmp").get_to(p.mlp_module_tmp);
        j.at("attn_module_tmp").get_to(p.attn_module_tmp);
        j.at("ln_f_module").get_to(p.ln_f_module);
        j.at("lm_head_module").get_to(p.lm_head_module);
        j.at("mom2_dataset").get_to(p.mom2_dataset);
        j.at("mom2_n_samples").get_to(p.mom2_n_samples);
        j.at("mom2_dtype").get_to(p.mom2_dtype);
        j.at("alg_name").get_to(p.alg_name);
        j.at("device").get_to(p.device);
        j.at("model_name").get_to(p.model_name);
        j.at("stats_dir").get_to(p.stats_dir);

        // Optional fields with defaults
        if (j.contains("max_length")) j.at("max_length").get_to(p.max_length);
        if (j.contains("model_parallel")) j.at("model_parallel").get_to(p.model_parallel);
        if (j.contains("fp16")) j.at("fp16").get_to(p.fp16);

        return p;
    }

    // Print function for debugging
    void print() const {
        std::cout << "ROMEHyperParams: alg_name=" << alg_name
                  << ", model_name=" << model_name
                  << ", layers=[";
        for (auto l : layers) std::cout << l << " ";
        std::cout << "]" << std::endl;
    }
};

int find_subsequence(
    Tensor &tokens,
    Tensor &target) {
    if (target.sequence() > tokens.sequence()) return -1;

    for (size_t i = 0; i <= tokens.sequence() - target.sequence(); ++i) {
        bool match = true;
        for (size_t j = 0; j < target.sequence(); ++j) {
            if (tokens.d<float>(0, i + j, 0, 0) != target.d<float>(0, j, 0, 0)) {
                match = false;
                break;
            }
        }
        if (match) return static_cast<int>(i);
    }

    return -1;
}

int main(int argc, char **argv) {
    std::ifstream f("../assets/rome_example.json");
    json sample_data = json::parse(f);
    // for (auto &item : sample_data) {
    //     std::cout << "Subject: " << item["subject"] << std::endl;
    //     std::cout << "Target New: " << item["target_new"] << std::endl;
    //     std::cout << "Prompt: " << item["prompt"] << std::endl;
    //     std::cout << "Ground Truth: ";
    //     for (const auto &gt : item["ground_truth"]) {
    //         std::cout << gt << " ";
    //     }
    //     std::cout << std::endl;
    // }

    json hyperParamJson = json::parse(std::ifstream("../assets/rome_hyper_param.json"));
    ROMEHyperParams hyperParams = ROMEHyperParams::from_json(hyperParamJson);
    hyperParams.print();

    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/qwen2.5_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/qwen2.5_merges.txt");
    cmdParser.add<string>("qnn-model", 'm', "specify mllm model path", false, "../models/Qwen2.5-1.5B-Instruct_rotated-noshadow.mllm");
    cmdParser.add<string>("decoding-model", '\0', "specify mllm model path", false, "../models/Qwen2.5-1.5B-Instruct_rotated-Q40.mllm");
    cmdParser.add<string>("billion", 'b', "[0.5B | 1.8B | 1.5B | [1.5B, 1.8B]-rotated]", false, "1.5B-rotated");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    const int chunk_size = 128; // Set the chunk size for the model

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("qnn-model");
    string decoding_model_path = cmdParser.get<string>("decoding-model");
    string model_billion = cmdParser.get<string>("billion");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = QWenTokenizer(vocab_path, merge_path);
    QWenNPUConfig config(tokens_limit, "1.5b-rotated", RoPEType::HFHUBROPE);
    auto model = rome_fwd::QWenForCausalLM_NPU(config, chunk_size);

    Context::Instance().initBackend(MLLM_QNN);

    model.load(model_path);
    // auto decoding_model = QWenForCausalLM(config);
    // decoding_model.load(decoding_model_path);

    mllm::optim::ZeroOrderOptimizer optimizer;

    for (int i = 0; i < sample_data.size(); ++i) {
        // auto input_str = tokenizer.apply_chat_template(in_strs[i]);
        std::string prompt = sample_data[i]["prompt"];
        std::string subject_str = sample_data[i]["subject"];
        std::string target_new_str = sample_data[i]["target_new"];

        std::string input_str = prompt + " " + target_new_str;

        // find substr starting 0 and ending with subject_str in prompt
        if (input_str.find(subject_str) == std::string::npos) {
            std::cerr << "Subject string not found in input string." << std::endl;
        }
        std::string prefix_str = input_str.substr(0, input_str.find(subject_str) + subject_str.length());
        std::cout << "Prefix: " << prefix_str << std::endl;
        auto prefix_token = tokenizer.tokenize(prefix_str, "prefix");

        int edit_idx = prefix_token.sequence() - 1;
        std::cout << "Edit index: " << edit_idx << std::endl;

        optimizer.setVectorIdx(edit_idx);

        auto pre_target_token = tokenizer.tokenize(prompt + " ", "pre_target");
        std::cout << "Pre-target token: " << prompt + " " << std::endl;

        int target_start_idx = pre_target_token.sequence();
        std::cout << "Target start index: " << target_start_idx << std::endl;

        std::cout << "Input: " << input_str << std::endl;
        auto [real_seq_length, input_tensor] = tokenizer.tokenizeWithPadding(input_str, chunk_size, config.vocab_size);

        int target_end_length = real_seq_length;
        std::cout << "Target end length: " << target_end_length << std::endl;

        vector<int> mock_target(chunk_size, -100);
        for (int i = target_start_idx; i < target_end_length; ++i) {
            mock_target[i] = (int)input_tensor.d<float>(0, i, 0, 0);
            std::cout << "Mock target[" << i << "]: " << mock_target[i] << std::endl;
        }

        std::cout << "[Q] " << input_str << std::endl;
        std::cout << "[A] " << std::flush;
        std::cout << "real_seq_length: " << real_seq_length << std::endl;

        // always turn on switching
        Context::Instance().inference_state().toggleSwitching();
        Context::Instance().inference_state().setExecutionType(PROMPT);

        // before training, run a forward to register the weights
        model({input_tensor});

        if (!std::filesystem::exists("qnn_context.bin")) {
            Context::Instance().globalBackends<QNNBackend>(MLLM_QNN)->saveQNNContext();
        }

        // freeze the QNN graph for inference, to avoid repeated tensor registration
        Context::Instance().inference_state().setQnnGraphFrozen(true);

        optimizer.initRandomVector();

        const int train_step = 5;

        for (int j = 0; j < train_step; j++) {
            // h-v forward
            // reset sequence length and execution type
            Context::Instance().inference_state().setCurSequenceLength(0);
            optimizer.applyPerturbation(mllm::optim::PERTUR_TYPE::ADD);
            auto result = model({input_tensor});

            auto loss_plus = optimizer.compute_nll_loss(result[0], mock_target);
            std::cout << "Loss Plus: " << loss_plus << std::endl;

            optimizer.removePerturbation(mllm::optim::PERTUR_TYPE::ADD);

            // h+v forward
            // reset sequence length and execution type
            Context::Instance().inference_state().setCurSequenceLength(0);
            optimizer.applyPerturbation(mllm::optim::PERTUR_TYPE::SUB);
            result = model({input_tensor});

            auto loss_minus = optimizer.compute_nll_loss(result[0], mock_target);
            std::cout << "Loss Minus: " << loss_minus << std::endl;

            optimizer.removePerturbation(mllm::optim::PERTUR_TYPE::SUB);

            optimizer.mobiedit_zero_order_optimization(loss_plus, loss_minus);
        }
    }
}
