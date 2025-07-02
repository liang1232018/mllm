/**
 * @brief Interface for managing inference-related state modules.
 *
 * The StateManager class defines a common interface for modules that manage
 * runtime state in a model inference engine. These states may include
 * sequence lengths, chunked prefill stages, speculative decoding information,
 * or any future task-specific metadata.
 *
 * The goal of this abstraction is to decouple backend execution from
 * stateful control logic, and to provide a unified mechanism for managing,
 * resetting, and debugging inference state across different components.
 *
 */

#pragma once

#include "Types.hpp"
#include <string>

namespace mllm {

class StateManager {
public:
    virtual ~StateManager() = default;

    virtual std::string name() const = 0;

    virtual void reset() = 0;
};

class InferenceStateManager : public StateManager {
public:
    std::string name() const override {
        return "InferenceStateManager";
    }
    void reset() override {
        execution_type_ = PROMPT;
        cur_sequence_length_ = 0;
        total_sequence_length_ = 0;
        is_switching_stage_ = false;
    }

    void setCurSequenceLength(int sequence_length) {
        cur_sequence_length_ = sequence_length;
    }
    int getCurSequenceLength() {
        return cur_sequence_length_;
    }
    void setTotalSequenceLength(int sequence_length) {
        total_sequence_length_ = sequence_length;
    }
    int getTotalSequenceLength() {
        return total_sequence_length_;
    }
    void toggleSwitching() {
        is_switching_stage_ = !is_switching_stage_;
    }
    void setChunkSize(int chunk_size) {
        chunk_size_ = chunk_size;
    }
    int getChunkSize() {
        return chunk_size_;
    }
    bool isStageSwitching() {
        return is_switching_stage_;
    }
    void setExecutionType(ExecutionType type) {
        execution_type_ = type;
    }
    ExecutionType getExecutionType() {
        return execution_type_;
    }

private:
    // indicate whether the state manager is in a prefill or decoding stage
    ExecutionType execution_type_ = PROMPT;
    // auto regression seq state
    int cur_sequence_length_ = 0;
    // total real seq length used for chunk & padding input
    int total_sequence_length_ = 0;
    // chunk size used in HeadLinear
    int chunk_size_ = 0;
    bool is_switching_stage_ = false;
};

class SpeculativeDecodingManager : public StateManager {
public:
    std::string name() const override {
        return "SpeculativeDecodingManager";
    }
    void reset() override {
        using_draft_ = false;
        last_draft_length_ = 0;
        last_verified_position_ids_.clear();
    }

    void setLastDraftLength(unsigned int draft_length) {
        last_draft_length_ = draft_length;
    }
    void setLastVerifiedPositionIds(const std::vector<unsigned int> &verified_position_ids) {
        last_verified_position_ids_ = verified_position_ids;
    }
    void setUsingDraft(bool _usingDraft) {
        this->using_draft_ = _usingDraft;
    }
    unsigned int getLastDraftLength() {
        return last_draft_length_;
    }
    std::vector<unsigned int> getLastVerifiedPositionIds() {
        return last_verified_position_ids_;
    }
    bool isUsingDraft() {
        return using_draft_;
    }

private:
    bool using_draft_ = false;
    std::vector<unsigned int> last_verified_position_ids_;
    unsigned int last_draft_length_ = 0;
};

} // namespace mllm
