#include "Context.hpp"
#include "QNNBackend.hpp"
#include <cstdlib>
#include <cstring>
#include "Types.hpp"
#include "cmdline.h"
#include "memory/MemInspect.hpp"
#include "models/qwen2_vl/configuration_qwen2_vl.hpp"
#include "models/qwen2_vl/modeling_npu_vit.hpp"
#include "models/qwen2_vl/modeling_qwen2_vl_npu.hpp"
#include "models/qwen2_vl/processing_qwen2_vl.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;
int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/qwen2vl_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/qwen2vl_merges.txt");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/qwen2_vl_vit_lm_rota_noshadow.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 1000);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    // TODO: add a function to calculate the chunk size
    const int chunk_size = 128;

    Context::Instance().initBackend(MLLM_QNN);

    ParamLoader param_loader(model_path);
    auto processor = Qwen2VLProcessor(vocab_path, merge_path);
    Qwen2VLNPUConfig config(tokens_limit, "1.5b-vl-rotated");

    auto prefill_embedding = test::Qwen2VL_ImagePatchAndEmbedding(config);
    prefill_embedding.load(model_path);

    vector<string> in_imgs = {
        "../assets/bus.png"};
    vector<string> in_strs = {
        "<|vision_start|><|image_pad|><|vision_end|>Imagine you are describing this image to someone who cannot see it. Explain everything you observe, including the background, subjects, their expressions, and any activities they appear to be doing.",
    };

    auto &in_str = in_strs[0];
    in_str = processor.tokenizer->apply_chat_template(in_str);
    auto input_tensors = processor.process(in_str, in_imgs[0]);

    const int real_seq_length = input_tensors[0].sequence();
    std::cout << "real seq length: " << real_seq_length << std::endl;

    const int num_iter = (real_seq_length + chunk_size - 1) / chunk_size;
    std::cout << "num_iter: " << num_iter << std::endl;


    auto merged_embd = prefill_embedding(input_tensors);

    prefill_embedding.get_position_ids(input_tensors, chunk_size);

    PRINT_MEMORY_USAGE("after qnn prefill embedding");

    Qwen2VLNPUConfig llm_config(tokens_limit, "1.5b-vl-rotated");
    auto prefill_body = Qwen2VL_PrefillBody(llm_config, chunk_size, llm_config.shadow_layers);
    prefill_body.load("../models/Qwen2-VL-2B-Instruct_vit_lm_rotated-Q40.mllm");

    auto merged_embd_warmup_tensor = Tensor(Context::Instance().globalBackends(MLLM_QNN));
    merged_embd_warmup_tensor.reshape(1, 1, chunk_size, 1536);
    merged_embd_warmup_tensor.setTtype(INPUT_TENSOR);
    merged_embd_warmup_tensor.alloc();

    merged_embd_warmup_tensor.setTtype(INPUT_TENSOR);
    input_tensors.back().setTtype(INPUT_TENSOR);
    vector<Tensor> prefill_input = {merged_embd_warmup_tensor, input_tensors.back()};
    prefill_body(prefill_input);
    std::cout << "after warm up" << std::endl;

    Module::isFirstChunk = false;
    Context::Instance().inference_state().setCurSequenceLength(0);
    Context::Instance().inference_state().setExecutionType(PROMPT);
    Context::Instance().inference_state().toggleSwitching();

    // set total seq length for HeadLinear execute, which can not get the real seq length from Opts
    Context::Instance().inference_state().setTotalSequenceLength(real_seq_length);
    // set chunk size for the HeadLinear execute, which can not get the chunk size from Opts
    Context::Instance().inference_state().setChunkSize(chunk_size);

    std::cout << "[Q] " << in_strs[0] << std::endl;
    std::cout << "[A] " << std::flush;

    for (auto &t : input_tensors) {
        t.setTtype(INPUT_TENSOR);
    }
    

    // 2. QNN LLM Prefill
    unsigned int out_token = 0;
    auto start_time = mllm_time_ms();
    int64_t prefill_time;
    for (auto i = 0; i < num_iter; ++i) {
        // copy the data from merged_embd[0] to merged_embd_warmup_tensor
        auto source = merged_embd[0].ptrAt<float>(0, 0, chunk_size * i, 0);
        auto dest = prefill_input[0].hostPtr<void>();
        if (i == 0) {
            memcpy(dest, source, std::min(prefill_input[0].cntSize(), merged_embd[0].cntSize()));
        } else {
            memcpy(dest, source, (merged_embd[0].sequence() % chunk_size) * merged_embd[0].dimension() * sizeof(float));
        }

        auto result = prefill_body(prefill_input);

        if (i == 0) { // turn off switching to avoid RoPE h_cnt_ reset to curSequenceLength in next chunk
            Context::Instance().inference_state().toggleSwitching();
        }

        if (i == num_iter - 1) {
            auto end_time = mllm_time_ms();
            prefill_time = end_time - start_time;

            auto outputs = processor.detokenize(result[0], real_seq_length % chunk_size);
            auto out_string = outputs.first;
            out_token = outputs.second;
            auto [not_end, output_string] = processor.tokenizer->postprocess(out_string);
            std::cout << output_string << std::flush;
            std::cout << "..." << std::flush;
        }
    }

    chatPostProcessing(out_token, input_tensors[0], {&input_tensors[1], &input_tensors[2]});


    std::cout << "--------AFTER QNN--------" << std::endl;

    // auto cpu_prefill_embedding = Qwen2VL_ImagePatchAndEmbedding(config);
    // cpu_prefill_embedding.load("../models/showui-2B-rotated.mllm");
    // merged_embd = cpu_prefill_embedding(input_tensors);

    if (!std::filesystem::exists("qnn_context.bin")) {
        Context::Instance().globalBackends<QNNBackend>(MLLM_QNN)->saveQNNContext();
    }

    return 0;
}