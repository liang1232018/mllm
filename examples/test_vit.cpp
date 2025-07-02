#include "Context.hpp"
#include <cstdlib>
#include <cstring>
#include "Types.hpp"
#include "cmdline.h"
#include "models/qwen2_vl/configuration_qwen2_vl.hpp"
#include "models/qwen2_vl/modeling_npu_vit.hpp"
#include "models/qwen2_vl/modeling_qwen2_vl_npu.hpp"
#include "models/qwen2_vl/processing_qwen2_vl.hpp"

using namespace mllm;
int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/showui_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/showui_merges.txt");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/showui2b-int8vit-test.mllm");
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
    const int chunk_size = 256;

    Context::Instance().initBackend(MLLM_QNN);

    ParamLoader param_loader(model_path);
    auto processor = Qwen2VLProcessor(vocab_path, merge_path);
    Qwen2VLNPUConfig config(tokens_limit, "1.5b-rotated");

    auto prefill_embedding = test::Qwen2VL_ImagePatchAndEmbedding(config);
    prefill_embedding.load(model_path);

    vector<string> in_imgs = {
        "../assets/bus.png"};
    vector<string> in_strs = {
        "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1.<|vision_start|><|image_pad|><|vision_end|>桌面",
    };

    auto &in_str = in_strs[0];
    in_str = processor.tokenizer->apply_chat_template(in_str);
    auto input_tensors = processor.process(in_str, in_imgs[0]);
    auto merged_embd = prefill_embedding(input_tensors);
    
    std::cout << "--------AFTER QNN--------" << std::endl;

    auto cpu_prefill_embedding = Qwen2VL_ImagePatchAndEmbedding(config);
    cpu_prefill_embedding.load("../models/showui-2B-rotated.mllm");
    merged_embd = cpu_prefill_embedding(input_tensors);

    return 0;
}