
#pragma once

#include "QnnCommon.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

namespace mllm {

#define CALL_QNN(apiCall)                                                \
    do {                                                                 \
        int errorCode = ((apiCall) & 0xFFFF);                            \
        if (errorCode != QNN_SUCCESS) {                                  \
            MLLM_LOG_ERROR("Error in file %s, line %d: error code %d\n", \
                           __FILE__, __LINE__, errorCode);               \
            assert(errorCode == QNN_SUCCESS);                            \
        }                                                                \
    } while (0)

// func def for loading QNN Interface
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t ***providerList,
                                                          uint32_t *numProviders);
// func def of loading QNN System Interface
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
    const QnnSystemInterface_t ***providerList, uint32_t *numProviders);

extern QnnInterfaceGetProvidersFn_t QnnInterface_getProviders;
extern QnnSystemInterfaceGetProvidersFn_t QnnSystemInterface_getProviders;

bool loadQNNSymbol();
bool loadQNNSystemSymbol();

} // namespace mllm