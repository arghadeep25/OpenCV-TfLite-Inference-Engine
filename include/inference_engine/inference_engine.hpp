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

#include <logging/log.hpp>
#include <tuple>

/**
    FLOAT32 = 0
    FLOAT16 = 1
    INT32 = 2
    UINT8 = 3
    INT64 = 4
    STRING = 5
    BOOL = 6
    INT16 = 7
    COMPLEX64 = 8
    INT8 = 9
 */
namespace tflite ::inference {
class TFLiteInferenceEngine {
 public:
  TFLiteInferenceEngine() = default;
  ~TFLiteInferenceEngine() = default;

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
   * @brief Infer the input image
   * @param input_image
   */
  std::tuple<TfLiteTensor *, TfLiteTensor *, TfLiteTensor *, TfLiteTensor *> infer(const cv::Mat &input_image) {
    if (input_image.empty())
      throw std::runtime_error("Input Image is empty");

    if (input_image.type() != CV_8UC3)
      throw std::runtime_error("Input Image is not in the format of CV_8UC3");

    if (!this->m_interpreter)
      throw std::runtime_error("Interpreter not initialized");

    auto *input = this->get_input_tensor();
    if (!input)
      throw std::runtime_error("Failed to get input tensor");

    memcpy(input, input_image.data,
           input_image.total() * input_image.elemSize());

    if (this->m_interpreter->Invoke() != kTfLiteOk)
      throw std::runtime_error("Failed to invoke the interpreter");

    const std::vector<int> &results = this->m_interpreter->outputs();
    std::cout << "Results: " << results.size() << std::endl;
    auto *output_locations = this->m_interpreter->tensor(results[0]);
    auto *output_classes = this->m_interpreter->tensor(results[1]);
    auto *output_scores = this->m_interpreter->tensor(results[2]);
    auto *num_detections = this->m_interpreter->tensor(results[3]);
    return {output_locations, output_classes, output_scores, num_detections};
  }

 public:
  /**
   * @brief Load the model from the given path in the memory and allocate
   * tensors
   * @param model_path Path to the model in the format of string
   */
  void load_model(const std::string &model_path) {
    // Load the model
    this->m_model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

    if (!this->m_model)
      throw std::runtime_error("Failed to load model: " + model_path);

    // Create the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*this->m_model, resolver)(&this->m_interpreter);

    if (!this->m_interpreter)
      throw std::runtime_error("Failed to create interpreter");

    if (static_cast<int>(get_num_threads()) == 0)
      throw std::runtime_error("Failed to get the number of threads");

    // Allocate tensor buffers and set the number of threads
    this->m_interpreter->SetNumThreads(static_cast<int>(get_num_threads()));
    if (this->m_interpreter->AllocateTensors() != kTfLiteOk)
      throw std::runtime_error("Failed to allocate tensors!");

    // Set the input and output details
    this->set_input_details();
    this->set_output_details();

    // Get the model details
    this->get_model_details();
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
      case kTfLiteUInt8:return this->m_interpreter->typed_input_tensor<uchar>(0);
      case kTfLiteFloat32:return this->m_interpreter->typed_input_tensor<float>(0);
      case kTfLiteInt8:return this->m_interpreter->typed_input_tensor<uchar>(0);
      default:throw std::runtime_error("Unsupported input tensor type");
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
    LOG("Input:: | Height:", this->m_input_height,
        " |\tWidth: ", this->m_input_width,
        " |\tChannels: ", this->m_input_channels, " |");
  }

 private:
  /**
   * @brief Get the output details
   */
  inline void get_output_details() const {
    LOG("Output:: | Height:", this->m_output_height,
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
