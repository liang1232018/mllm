#pragma once

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

private:
    friend class Backend;

    Context();
    ~Context() = default;

    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

};

} // namespace mllm