/**
* @file test_inference.hpp
* @details Test cases for inference engine
* @author Arghadeep Mazumder
* @version 0.1.0
* @copyright -
*/

#include "infer/infer.hpp"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

using namespace tflite::inference;

class TFLiteInferenceEngineTest : public ::testing::Test {
protected:
  TFLiteInferenceEngine engine;
  std::string model_path =
      std::string(PROJECT_SOURCE_DIR) + "/models/mobilenet_ssd_v1.tflite";

  // Input
  const int input_height = 300;
  const int input_width = 300;
  const int input_channels = 3;
  // Output
  const int output_height = 10;
  const int output_width = 4;
  const int output_channels = 0;
};

TEST_F(TFLiteInferenceEngineTest, GetInputHeightReturnsCorrectValue) {
  engine.load_model(this->model_path);
  EXPECT_EQ(engine.get_input_height(), this->input_height);
}

TEST_F(TFLiteInferenceEngineTest, GetInputHeightReturnsWrongValue) {
  engine.load_model(this->model_path);
  EXPECT_NE(engine.get_input_height(), 0);
}

TEST_F(TFLiteInferenceEngineTest, GetInputWidthReturnsCorrectValue) {
  engine.load_model(this->model_path);
  EXPECT_EQ(engine.get_input_width(), this->input_width);
}

TEST_F(TFLiteInferenceEngineTest, GetInputWidthReturnsWrongValue) {
  engine.load_model(this->model_path);
  EXPECT_NE(engine.get_input_width(), 0);
}

TEST_F(TFLiteInferenceEngineTest, GetInputChannelsReturnsCorrectValue) {
  engine.load_model(this->model_path);
  EXPECT_EQ(engine.get_input_channels(), this->input_channels);
}

TEST_F(TFLiteInferenceEngineTest, GetInputChannelsReturnsWrongValue) {
  engine.load_model(this->model_path);
  EXPECT_NE(engine.get_input_channels(), 0);
}

TEST_F(TFLiteInferenceEngineTest, GetOutputHeightReturnsCorrectValue) {
  engine.load_model(this->model_path);
  EXPECT_EQ(engine.get_output_height(), this->output_height);
}

TEST_F(TFLiteInferenceEngineTest, GetOutputHeightReturnsWrongValue) {
  engine.load_model(this->model_path);
  EXPECT_NE(engine.get_output_height(), 0);
}

TEST_F(TFLiteInferenceEngineTest, GetOutputWidthReturnsCorrectValue) {
  engine.load_model(this->model_path);
  EXPECT_EQ(engine.get_output_width(), this->output_width);
}

TEST_F(TFLiteInferenceEngineTest, GetOutputWidthReturnsWrongValue) {
  engine.load_model(this->model_path);
  EXPECT_NE(engine.get_output_width(), 0);
}

TEST_F(TFLiteInferenceEngineTest, GetOutputChannelsReturnsCorrectValue) {
  engine.load_model(this->model_path);
  EXPECT_EQ(engine.get_output_channels(), this->output_channels);
}

TEST_F(TFLiteInferenceEngineTest, GetOutputChannelsReturnsWrongValue) {
  engine.load_model(this->model_path);
  EXPECT_NE(engine.get_output_channels(), 1000);
}

TEST_F(TFLiteInferenceEngineTest, InferReturnsNullptrsForEmptyImage) {
  cv::Mat empty_image;
  auto result = engine.infer(empty_image);
  EXPECT_EQ(std::get<0>(result), nullptr);
  EXPECT_EQ(std::get<1>(result), nullptr);
  EXPECT_EQ(std::get<2>(result), nullptr);
  EXPECT_EQ(std::get<3>(result), nullptr);
}

TEST_F(TFLiteInferenceEngineTest, InferReturnsNullptrsForInvalidImageType) {
  cv::Mat invalid_image(224, 224, CV_8UC1);
  auto result = engine.infer(invalid_image);
  EXPECT_EQ(std::get<0>(result), nullptr);
  EXPECT_EQ(std::get<1>(result), nullptr);
  EXPECT_EQ(std::get<2>(result), nullptr);
  EXPECT_EQ(std::get<3>(result), nullptr);
}

TEST_F(TFLiteInferenceEngineTest, InferReturnsValidTensorsForValidImage) {
  engine.load_model(this->model_path);
  cv::Mat valid_image(224, 224, CV_8UC3, cv::Scalar(0, 0, 0));
  auto result = engine.infer(valid_image);
  EXPECT_NE(std::get<0>(result), nullptr);
  EXPECT_NE(std::get<1>(result), nullptr);
  EXPECT_NE(std::get<2>(result), nullptr);
  EXPECT_NE(std::get<3>(result), nullptr);
}

TEST_F(TFLiteInferenceEngineTest, LoadModelReturnsErrorForInvalidPath) {
  auto status = engine.load_model("invalid/path/to/model.tflite");
  EXPECT_EQ(status, tflite::inference::InferenceStatus::MODEL_LOAD_ERROR);
}

TEST_F(TFLiteInferenceEngineTest, LoadModelReturnsOkForValidPath) {
  auto status = engine.load_model(this->model_path);
  EXPECT_EQ(status, tflite::inference::InferenceStatus::SUCCESS);
}

TEST_F(TFLiteInferenceEngineTest, LoadModelReturnsErrorForEmptyPath) {
  auto status = engine.load_model("");
  EXPECT_EQ(status, tflite::inference::InferenceStatus::MODEL_LOAD_ERROR);
}
