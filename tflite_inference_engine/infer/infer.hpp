/**
* @file inference.hpp
* @details Inferencing engine for TensorFlow Lite models
* @author Arghadeep Mazumder
* @version 0.1.0
* @copyright -
*/

#ifndef INFERENCE_ENGINE_HPP
#define INFERENCE_ENGINE_HPP

#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#include <filesystem>
#include <log/log.hpp>
#include <tuple>
#include <utils/inference_status.hpp>

namespace tflite ::inference {
class TFLiteInferenceEngine {
public:
  TFLiteInferenceEngine() = default;
  ~TFLiteInferenceEngine() = default;

  TFLiteInferenceEngine(const TFLiteInferenceEngine &) = delete;
  TFLiteInferenceEngine &operator=(const TFLiteInferenceEngine &) = delete;
  TFLiteInferenceEngine(TFLiteInferenceEngine &&) = delete;
  TFLiteInferenceEngine &operator=(TFLiteInferenceEngine &&) = delete;

public:
  /**
   * @brief Get the input height
   * @return Input height
   */
  [[nodiscard]] int get_input_height() const { return this->m_input_height; }

  /**
   * @brief Get the input width
   * @return Input width
   */
  [[nodiscard]] int get_input_width() const { return this->m_input_width; }

  /**
   * @brief Get the input channels
   * @return Input channels
   */
  [[nodiscard]] int get_input_channels() const {
    return this->m_input_channels;
  }

public:
  /**
   * @brief Get the output height
   * @return Output height
   */
  [[nodiscard]] int get_output_height() const { return this->m_output_height; }

  /**
   * @brief Get the output width
   * @return Output width
   */
  [[nodiscard]] int get_output_width() const { return this->m_output_width; }

  /**
   * @brief Get the output channels
   * @return Output channels
   */
  [[nodiscard]] int get_output_channels() const {
    return this->m_output_channels;
  }

public:
  /**
   * @brief Get the inference results from the given input image
   * @param input_image Input image in the format of cv::Mat
   * @return Tuple of output locations, output classes, output scores
   *         and number of detections
   */
  std::tuple<float *, float *, float *, float *>
  infer(const cv::Mat &input_image) {
    if (input_image.empty()) {
      LOG_ERROR("Input image is empty");
      return {nullptr, nullptr, nullptr, nullptr};
    }

    if (!this->m_interpreter) {
      LOG_ERROR("Interpreter not initialized");
      return {nullptr, nullptr, nullptr, nullptr};
    }

    auto *input = this->get_input_tensor();
    if (!input) {
      LOG_ERROR("Failed to get input tensor");
      return {nullptr, nullptr, nullptr, nullptr};
    }

    memcpy(input, input_image.data,
           input_image.total() * input_image.elemSize());

    if (this->m_interpreter->Invoke() != kTfLiteOk) {
      LOG_ERROR("Failed to invoke the interpreter");
      return {nullptr, nullptr, nullptr, nullptr};
    }

    if (this->m_interpreter->outputs().size() != 4) {
      LOG_ERROR("Output size is not equal to 4");
      return {nullptr, nullptr, nullptr, nullptr};
    }

    if (this->m_interpreter->typed_output_tensor<float>(0) == nullptr ||
        this->m_interpreter->typed_output_tensor<float>(1) == nullptr ||
        this->m_interpreter->typed_output_tensor<float>(2) == nullptr ||
        this->m_interpreter->typed_output_tensor<float>(3) == nullptr) {
      LOG_ERROR("Output tensor is nullptr");
      return {nullptr, nullptr, nullptr, nullptr};
    }

    return {this->m_interpreter->typed_output_tensor<float>(0),
            this->m_interpreter->typed_output_tensor<float>(1),
            this->m_interpreter->typed_output_tensor<float>(2),
            this->m_interpreter->typed_output_tensor<float>(3)};
  }

public:
  /**
   * @brief Load the model from the given path in the memory and allocate
   * tensors
   * @param model_path Path to the model in the format of string
   */
  inference::InferenceStatus load_model(const std::string &model_path) {
    if (model_path.empty() || !std::filesystem::exists(model_path)) {
      LOG_ERROR("Model path is empty or does not exist");
      return inference::InferenceStatus::MODEL_LOAD_ERROR;
    }
    // Load the model
    this->m_model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

    if (!this->m_model) {
      LOG_ERROR("Failed to load model: ", model_path);
      return inference::InferenceStatus::MODEL_LOAD_ERROR;
    }

    // Create the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*this->m_model, resolver)(&this->m_interpreter);

    if (!this->m_interpreter) {
      LOG_ERROR("Failed to create interpreter");
      return inference::InferenceStatus::INTERPRETER_ERROR;
    }

    if (static_cast<int>(get_num_threads()) == 0) {
      LOG_ERROR("Failed to get the number of threads");
      return inference::InferenceStatus::INVOCATION_ERROR;
    }

    // Allocate tensor buffers and set the number of threads
    this->m_interpreter->SetNumThreads(static_cast<int>(get_num_threads()));
    if (this->m_interpreter->AllocateTensors() != kTfLiteOk) {
      LOG_ERROR("Failed to allocate tensors");
      return inference::InferenceStatus::TENSOR_ALLOCATION_ERROR;
    }

    // Set the input and output details
    this->set_input_details();
    this->set_output_details();

    // Get the model details
    this->get_model_details();
    return inference::InferenceStatus::SUCCESS;
  }

private:
  /**
   * @brief Get the input tensor
   * @return Input tensor
   */
  void *get_input_tensor() {
    int tensor_type =
        this->m_interpreter->tensor(this->m_interpreter->inputs()[0])->type;
    switch (tensor_type) {
    case kTfLiteUInt8:
      return this->m_interpreter->typed_input_tensor<uchar>(0);
    case kTfLiteFloat32:
      return this->m_interpreter->typed_input_tensor<float>(0);
    case kTfLiteInt8:
      return this->m_interpreter->typed_input_tensor<uchar>(0);
    default:
      throw std::runtime_error("Unsupported input tensor type");
    }
  }

private:
  /**
   * @brief Set the input details
   */
  void set_input_dims_array() {
    this->m_input_dims =
        this->m_interpreter->tensor(this->m_interpreter->inputs()[0])->dims;
  }

  /**
   * @brief Set the input height
   */
  void set_input_height() {
    this->m_input_height = this->m_input_dims->data[1];
  }

  /**
   * @brief Set the input width
   */
  void set_input_width() { this->m_input_width = this->m_input_dims->data[2]; }

  /**
   * @brief Set the input channels
   */
  void set_input_channels() {
    this->m_input_channels = this->m_input_dims->data[3];
  }

private:
  /**
   * @brief Set the output details
   */
  void set_output_dims_array() {
    this->m_output_dims =
        this->m_interpreter->tensor(this->m_interpreter->outputs()[0])->dims;
  }

  /**
   * @brief Set the output height
   */
  void set_output_height() {
    this->m_output_height = this->m_output_dims->data[1];
  }

  /**
   * @brief Set the output width
   */
  void set_output_width() {
    this->m_output_width = this->m_output_dims->data[2];
  }

  /**
   * @brief Set the output channels
   */
  void set_output_channels() {
    this->m_output_channels = this->m_output_dims->data[3];
  }

private:
  /**
   * @brief Set the input details
   */
  void set_input_details() {
    this->set_input_dims_array();
    this->set_input_height();
    this->set_input_width();
    this->set_input_channels();
  }

private:
  /**
   * @brief Set the output details
   */
  void set_output_details() {
    this->set_output_dims_array();
    this->set_output_height();
    this->set_output_width();
    this->set_output_channels();
  }

private:
  /**
   * @brief Get the model details
   */
  inline void get_model_details() const {
    this->get_input_details();
    this->get_output_details();
  }

private:
  /**
   * @brief Get the input details
   */
  inline void get_input_details() const {
    LOG_INFO("Input:: | Height:", this->m_input_height,
             " |\tWidth: ", this->m_input_width,
             " |\tChannels: ", this->m_input_channels, " |");
  }

private:
  /**
   * @brief Get the output details
   */
  inline void get_output_details() const {
    LOG_INFO("Output:: | Height:", this->m_output_height,
             " |\tWidth: ", this->m_output_width,
             " |\tChannels: ", this->m_output_channels, " |");
  }

private:
  /**
   * @brief Get the number of threads
   * @return Number of threads
   */
  static unsigned int get_num_threads() {
    return std::thread::hardware_concurrency();
  }

private:
  int m_input_height{};
  int m_input_width{};
  int m_input_channels{};

  int m_output_height{};
  int m_output_width{};
  int m_output_channels{};

  TfLiteIntArray *m_input_dims{};
  TfLiteIntArray *m_output_dims{};

  std::unique_ptr<tflite::Interpreter> m_interpreter;
  std::unique_ptr<tflite::FlatBufferModel> m_model;
};

} // namespace tflite::inference

#endif // INFERENCE_ENGINE_HPP
