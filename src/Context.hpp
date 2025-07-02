#pragma once

#include "StateManager.hpp"
#include "Types.hpp"
#include "Backend.hpp"

namespace mllm {

class Context {
public:
    static Context &Instance();

    Backend *globalBackends(BackendType type) const {
        return Backend::global_backends[type];
    }

    void initBackend(BackendType type);

    InferenceStateManager& inference_state() {
        return inference_state_;
    }

    SpeculativeDecodingManager& speculative_decoding_state() {
        return speculative_decoding_state_;
    }

private:
    friend class Backend;

    Context();
    ~Context() = default;

    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    InferenceStateManager inference_state_;
    SpeculativeDecodingManager speculative_decoding_state_;
};

} // namespace mllm