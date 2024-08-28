//
// Created by arghadeep on 22.08.24.
//

#ifndef INFERENCE_STATUS_HPP
#define INFERENCE_STATUS_HPP

namespace tflite::inference {
enum class InferenceStatus {
  SUCCESS,
  MODEL_LOAD_ERROR,
  INTERPRETER_ERROR,
  TENSOR_ALLOCATION_ERROR,
  INVOCATION_ERROR,
  INPUT_ERROR
};
} // namespace tflite::inference

#endif // INFERENCE_STATUS_HPP
