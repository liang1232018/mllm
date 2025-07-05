#include "Context.hpp"
#include <cstdlib>
#include <cstring>
#include "Types.hpp"
#include "cmdline.h"
#include "memory/MemInspect.hpp"
#include "models/qwen2_vl/configuration_qwen2_vl.hpp"
#include "models/qwen2_vl/modeling_npu_vit.hpp"
#include "models/qwen2_vl/modeling_qwen2_vl_npu.hpp"
#include "models/qwen2_vl/processing_qwen2_vl.hpp"

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

    std::cout << "--------AFTER QNN--------" << std::endl;

    // auto cpu_prefill_embedding = Qwen2VL_ImagePatchAndEmbedding(config);
    // cpu_prefill_embedding.load("../models/showui-2B-rotated.mllm");
    // merged_embd = cpu_prefill_embedding(input_tensors);

    return 0;
}