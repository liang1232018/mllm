#include "QNNUtils.hpp"
#include "Log.h"
#include <dlfcn.h>

namespace mllm {

QnnInterfaceGetProvidersFn_t QnnInterface_getProviders = nullptr;

bool loadQNNSymbol() {
    MLLM_LOG_INFO_STREAM << "QNN Backend: libQnnHtp.so" << std::endl;
    void *qnnLibHandle = nullptr;
    qnnLibHandle = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
    const char *errorOpen = dlerror();
    if (!qnnLibHandle) {
        MLLM_LOG_ERROR_LEGACY("Failed to open QNN libs. Ensure that the libs related to the QNN HTP backend is available in your environment. dlerror() returns %s.\n", errorOpen);
        return false;
    }

    QnnInterface_getProviders = (QnnInterfaceGetProvidersFn_t)dlsym(qnnLibHandle, "QnnInterface_getProviders");
    const char *errorSym = dlerror();
    if (!QnnInterface_getProviders) {
        MLLM_LOG_ERROR_LEGACY("Failed to load symbol <QnnInterface_getProviders>. dlerror returns %s.\n", errorSym);
        dlclose(qnnLibHandle);
        return false;
    }

    return true;
}

QnnSystemInterfaceGetProvidersFn_t QnnSystemInterface_getProviders = nullptr;

bool loadQNNSystemSymbol() {
    void *systemLibraryHandle = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
    const char *errorOpen = dlerror();
    if (!systemLibraryHandle) {
        MLLM_LOG_ERROR_LEGACY("Failed to open QNN System libs. Ensure that the libs related to the QNN System backend is available in your environment. dlerror() returns %s.\n", errorOpen);
        return false;
    }

    QnnSystemInterface_getProviders = (QnnSystemInterfaceGetProvidersFn_t)dlsym(systemLibraryHandle, "QnnSystemInterface_getProviders");
    const char *errorSym = dlerror();
    if (!QnnSystemInterface_getProviders) {
        MLLM_LOG_ERROR_LEGACY("Failed to load symbol <QnnSystemInterface_getProviders>. dlerror returns %s.\n", errorSym);
        dlclose(systemLibraryHandle);
        return false;
    }

    return true;
}

} // namespace mllm