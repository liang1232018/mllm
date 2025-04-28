#include <cstdlib>
#include <cstring>
#include <iostream>
#include "QNNBackend.hpp"
#include "Timing.hpp"
#include "Types.hpp"
#include "cmdline.h"
#include "models/qwen2_vl/configuration_qwen2_vl.hpp"
#include "models/qwen2_vl/modeling_qwen2_vl_npu.hpp"
#include "models/qwen2_vl/processing_qwen2_vl.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;
int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/qwen2vl_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/qwen2vl_merges.txt");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/qwen2-vl-w8-i8bias-128-xdl-test.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 1000);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("model");
    const string cpu_model_path = "../models/qwen-2-vl-2b-instruct-q4_k.mllm";
    int tokens_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");
    // NOTE: this chunk size is only for bus.png
    // TODO: add a function to calculate the chunk size
    const int chunk_size = 65;

    Module::initBackend(MLLM_QNN);

    ParamLoader param_loader(model_path);
    auto processor = Qwen2VLProcessor(vocab_path, merge_path);
    Qwen2VLConfig config(tokens_limit, "1.5b");
    auto model_config = Qwen2VLConfig(config);

    auto prefill_embedding = Qwen2VL_ImagePatchAndEmbedding(config);
    auto prefill_body = Qwen2VL_PrefillBody(config, chunk_size);
    prefill_embedding.load(cpu_model_path);
    prefill_body.load(model_path);

    auto decoding_model = Qwen2VL_Decoding_Model(model_config);
    decoding_model.load(cpu_model_path);

    vector<string> in_imgs = {
        "../assets/bus.png"};
    vector<string> in_strs = {
        "<|vision_start|><|image_pad|><|vision_end|>Describe this image.",
    };

    auto &in_str = in_strs[0];
    in_str = processor.tokenizer->apply_chat_template(in_str);
    auto input_tensors = processor.process(in_str, in_imgs[0]);
    prefill_embedding.get_position_ids(input_tensors);

    // warm up
    auto merged_embd_warmup_tensor = Tensor(Backend::global_backends[MLLM_QNN]);
    merged_embd_warmup_tensor.reshape(1, 1, chunk_size, 1536);
    merged_embd_warmup_tensor.setTtype(INPUT_TENSOR);
    merged_embd_warmup_tensor.alloc();
    prefill_body({merged_embd_warmup_tensor, input_tensors.back()});

    std::cout << "after warm up" << std::endl;

    Module::isFirstChunk = false;
    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setCurSequenceLength(0);
    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(PROMPT);
    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();

    for (auto &t : input_tensors) {
        t.setTtype(INPUT_TENSOR);
    }

    // 1. get the vit embedding using CPU
    auto merged_embd = prefill_embedding(input_tensors);

    // 2. QNN LLM Prefill
    auto start_time = mllm_time_ms();

    merged_embd_warmup_tensor.setTtype(INPUT_TENSOR);
    input_tensors.back().setTtype(INPUT_TENSOR);
    // copy the data from merged_embd[0] to merged_embd_warmup_tensor
    auto source = merged_embd[0].hostPtr<void>();
    auto dest = merged_embd_warmup_tensor.hostPtr<void>();
    memcpy(dest, source, merged_embd[0].cntSize());

    auto result = prefill_body({merged_embd_warmup_tensor, input_tensors.back()});
    auto end_time = mllm_time_ms();
    std::cout << end_time - start_time << " ms" << std::endl;

    auto outputs = processor.detokenize(result[0]);
    auto out_string = outputs.first;
    auto out_token = outputs.second;
    auto [not_end, output_string] = processor.tokenizer->postprocess(out_string);
    std::cout << output_string << std::flush;

    chatPostProcessing(out_token, input_tensors[0], {&input_tensors[1], &input_tensors[2]});

    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setCurSequenceLength(65);
    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(AUTOREGRESSIVE);

    // 3. CPU LLM Decoding
    for (auto &t : input_tensors) {
        t.setTtype(INPUT_TENSOR);
    }
    for (int step = 0; step < 100; step++) {
        prefill_embedding.get_position_ids(input_tensors);

        auto result = decoding_model(input_tensors);
        auto outputs = processor.detokenize(result[0]);
        auto out_string = outputs.first;
        auto out_token = outputs.second;
        auto [not_end, output_string] = processor.tokenizer->postprocess(out_string);
        if (!not_end) { break; }
        std::cout << output_string << std::flush;
        chatPostProcessing(out_token, input_tensors[0], {&input_tensors[1], &input_tensors[2]});

        if (step == 0) static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
    }

    std::cout << std::endl;

    if (!std::filesystem::exists("qnn_context.bin")) {
        static_cast<QNNBackend *>(Backend::global_backends[MLLM_QNN])->saveQNNContext();
    }

    return 0;
}