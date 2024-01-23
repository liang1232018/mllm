[![Website](https://img.shields.io/badge/website-visit-green)](https://ubiquitouslearning.github.io/mllm_website/)
[![Documentation](https://img.shields.io/badge/view-docs-blue)](https://ubiquitouslearning.github.io/mllm_website/introduction/getstarted/)
[![Actions Status](https://github.com/UbiquitousLearning/mllm/workflows/Tests/badge.svg)](https://github.com/UbiquitousLearning/mllm/actions)

**mllm** is a fast and lightweight <ins>multimodal LLM</ins> inference engine for mobile and edge devices.

- Plain C/C++ implementation without dependencies
- Optimized for multimodal LLMs like fuyu-8B
- Supported: ARM NEON and x86 AVX2
- 4-bit and 6-bit integer quantization

Wait.. why on-device multimodal LLM? - It's a key building block for [intelligent personal agent](https://arxiv.org/pdf/2401.05459.pdf), text-based image searching/retrieval, screen VQA, and many more exciting mobile apps, without giving away your private data (chat history, screenshots, taken photos, etc).


### Contents
- [Android Demo](#android-demo)
- [Support models](#support-models)
- [Quick Start](#quick-start)
    - [Get the Code](#get-the-code)
    - [Check prerequisites](#check-prerequisites)
    - [Run for Android](#run-for-android)
    - [Run for Linux](#run-for-linux)
- [Customization](#customization)
    - [Convert models](#convert-models)
    - [Convert Vocabulary](#convert-vocabulary)
    - [Quantize models](#quantize-models)
- [Roadmap](#roadmap)
- [Documentation](#documentation)
- [Contribution](#contribution)
- [Acknowledgments](#acknowledgments)
- [License](#license)


## Android Demo

https://github.com/lx200916/mllm/assets/44310445/4ede7e6e-f3d2-4ef5-bd32-73c022edfdbe

## Support models

[//]: # (* ✔️ : Support and test well on mobile devices.)

[//]: # ()
[//]: # (* ⚠️ : It is under construction or still has some bugs.)

[//]: # ()
[//]: # (* ❌  : Not support yet.)

|                          | FP32 | INT4 |
|--------------------------|-----|------|
| LLaMA-1/2 7B             | ✔️  | ✔️   |
| Alpaca 7B                | ✔️  | ✔️   |
| TinyLLaMA 1.1B           | ✔️  | ✔️   |
| Persimmon 8B             | ✔️  | ✔️   |
| Fuyu 8B                  | ✔️  | ✔️   |
| Vision Transformer       | ✔️  | ✔️   |
| CLIP                     | ✔️  | ✔️   |
| ImageBind (3 modalities) | ✔️  | ✔️   |
| LLaVA 7B                 | ✔️  | ✔️   |

## Quick Start

### Get the Code

```bash
git clone https://github.com/UbiquitousLearning/mllm
cd mllm
```

### Check prerequisites

Building mllm requires following tools:

- gcc(11.4+) / clang (11.0+)
- CMake >= 3.18
- Android NDK Toolchains >= 26

### Try it on Android

#### Build

  ```bash
  export ANDROID_NDK=/path/to/your/ndk
  cd scripts
  ./build_android.sh
  ```

#### Run Fuyu-8B

Download the model from [here](https://huggingface.co/mllmTeam), or using the following instructions

```bash
mkdir ../models && cd ../models
# Download fuyu-8b-q4_k.mllm
wget https://huggingface.co/mllmTeam/fuyu-8b-mllm/blob/resolve/main/fuyu-8b-q4_k.mllm?download=true  -O fuyu-8b-q4_k.mllm
```

Run on an android phone with at least 12GB of memory.

```bash
cd ../script
./run_fuyu.sh
```

Result are as followed:

```
> ./main_fuyu
Load model: 6.34905 s
[Q] [../assets/bus.png]Generate a coco-style caption.

[A]  A blue bus driving down▁the▁road next▁to a sidewalk.
[Q] [../assets/two_cats.jpg]What's this?

[A]  Two cats laying on▁top▁of a pink blanket.
```

#### Run LLaMA-2-7B

Download model

```bash
mkdir ../models && cd ../models
# Dowload llama-2-7b-chat-q4_k.mllm
wget https://huggingface.co/mllmTeam/llama-2-7b-mllm/resolve/main/llama-2-7b-chat-q4_k.mllm?download=true  -O llama-2-7b-chat-q4_k.mllm
```

Run on an android phone with at least 12GB of memory.

```bash
cd ../script
./run_llama.sh
```

Result are as followed:

```
> ./main_llama
Load model: 10.0351 s
[Q]  Hello, who are you?
[A]

I am a bot, I'm here to help answer your questions and provide information. I'm just an AI, I don't have personal experiences or feelings like humans do, but I'm here to help you with any questions you may have. Is there something specific you would like to know or talk about?
[Q]  What can you do?
[A]
I can help with a wide range of tasks, including:
  
* Answering questions on various topics, such as science, history, technology, and more.
* Providing information on different subjects, such as culture, entertainment, and lifestyle.
* Helping with language-related tasks, such as translating text or improving grammar and syntax.
* Generating text, such as articles, stories, or even entire books.
* Assisting with coding
[Q]  Please introduce Beijing University of Posts and Telecommunications.
[A]
Beijing University of Posts and Telecommunications (BUPT) is a public research university located in Beijing, China. It was established in 1955 and is one of the top universities in China, with a strong focus on engineering, technology, and telecommunications.
BUPT offers a wide range of undergraduate and graduate programs in fields such as computer science, electrical engineering, software engineering, and communication engineering. The university
```

#### Run ImageBind

Download model

```bash
mkdir ../models && cd ../models
# Download imagebind_huge-q4_k.mllm
wget https://huggingface.co/mllmTeam/imagebind_huge-mllm/resolve/blob/main/imagebind_huge-q4_k.mllm?download=true -O imagebind_huge-q4_k.mllm 
```

Run on an android phone with at least 4GB of memory.

```bash
cd ../script
./run_imagebind.sh
```

Result are as followed:
```
> ./main_imagebind 
Load model: 2.17264 s
vision X text :
0.9986231 0.0013304 0.0000464
0.0000365 0.9998704 0.0000931
0.0000124 0.0085099 0.9914777
vision X audio :  
0.8089266 0.1228399 0.0682334 
0.0706672 0.8396024 0.0897304
0.0021378 0.0015886 0.9962736
```


### Run for Linux

#### Build
```bash
cd scripts
./build.sh
 ```

#### Run Fuyu-8B

```bash
cd ./bin
./main_fuyu -m ../models/fuyu-8b-q4_k.mllm -v ../vacob/fuyu_vocab.mllm
 ```

#### Run LLaMA-2-7B

```bash
cd ./bin
./main_llama -m ../models/llama-2-7b-chat-q4_k.mllm -v ../vacob/llama_vocab.mllm
```


#### Run ImageBind

```bash
cd ./bin
./main_imagebind -m ../models/imagebind_huge-q4_k.mllm -v ../vacob/clip_vocab.mllm
```


## Customization

### Convert models

You can download models from [here](https://huggingface.co/mllmTeam), or you can convert a pytorch/safetensor model to
mllm model by yourself.

```bash
cd tools/convertor
pip install -r ./requirements.txt

# for one file pytorch model
python convert.py --input_model=model.pth --output_model=model.mllm --type=torch

# for multi-file pytorch model
python convert.py --input_model=pytorch_model.bin.index.json --output_model=model.mllm --type=torch

# for one file safetensor model
python convert.py --input_model=model.bin --output_model=model.mllm --type=safetensor

# for multi-file safetensor model
python convert.py --input_model=model.safetensors.index.json --output_model=model.mllm --type=safetensor
``` 

### Convert vocabulary

You can convert vocabulary to mllm vocabulary as followed.

```bash
cd tools/convertor
python vocab.py --input_file=tokenizer.json --output_file=vocab.mllm --type=Unigram
```

### Quantize models

You can quantize mllm model to int4 model by yourself.
mllm only support two quantize modes: Q4_0 and Q4_K.

```bash
cd bin
./quantize model.mllm model_q4_0.mllm Q4_K
```

## Roadmap

- More backends like QNN
- More models like LLaVA
- More optimizations like LUT-GEMM
- [More..](https://ubiquitouslearning.github.io/mllm_website/roadmap/roadmap/)

## Documentation

See the [documentation](https://ubiquitouslearning.github.io/mllm_website/introduction/getstarted/) here for more
information

## Contribution

Read the [contribution](https://ubiquitouslearning.github.io/mllm_website/contributing/contributing/) before you
contribute.

## Acknowledgments

mllm reuses many low-level kernel implementation from [ggml](https://github.com/ggerganov/ggml) on ARM CPU.
It also utilizes [stb](https://github.com/nothings/stb) and [wenet](https://github.com/wenet-e2e/wenet) for
pre-processing images and audios.
mllm also has benefitted from following projects: [llama.cpp](https://github.com/ggerganov/llama.cpp)
and [MNN](https://github.com/alibaba/MNN).

## License

### Overall Project License

This project is licensed under the terms of the MIT License. Please see the [LICENSE](LICENSE) file in the root
directory for the full text of the MIT License.

### Apache 2.0 Licensed Components

Certain component([wenet](https://github.com/wenet-e2e/wenet)) of this project is licensed under the Apache License 2.0.
These component is clearly identified in their respective subdirectories along with a copy of the Apache License 2.0.
For the full text of the Apache License 2.0, please refer to the [LICENSE-APACHE](third_party/wenet_audio/LICENSE) file
located in the relevant subdirectories.
