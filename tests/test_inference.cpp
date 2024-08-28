//
// Created by arghadeep on 22.08.24.
//
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "infer/infer.hpp"

using namespace tflite::inference;

class TFLiteInferenceEngineTest : public ::testing::Test {
protected:
  TFLiteInferenceEngine engine;
};

TEST_F(TFLiteInferenceEngineTest, GetInputHeightReturnsCorrectValue) {
  engine.load_model("path/to/model.tflite");
  EXPECT_EQ(engine.get_input_height(), 224); // Assuming model input height is 224
}

TEST_F(TFLiteInferenceEngineTest, GetInputWidthReturnsCorrectValue) {
  engine.load_model("path/to/model.tflite");
  EXPECT_EQ(engine.get_input_width(), 224); // Assuming model input width is 224
}

TEST_F(TFLiteInferenceEngineTest, GetInputChannelsReturnsCorrectValue) {
  engine.load_model("path/to/model.tflite");
  EXPECT_EQ(engine.get_input_channels(), 3); // Assuming model input channels is 3
}

TEST_F(TFLiteInferenceEngineTest, GetOutputHeightReturnsCorrectValue) {
  engine.load_model("path/to/model.tflite");
  EXPECT_EQ(engine.get_output_height(), 7); // Assuming model output height is 7
}

TEST_F(TFLiteInferenceEngineTest, GetOutputWidthReturnsCorrectValue) {
  engine.load_model("path/to/model.tflite");
  EXPECT_EQ(engine.get_output_width(), 7); // Assuming model output width is 7
}

TEST_F(TFLiteInferenceEngineTest, GetOutputChannelsReturnsCorrectValue) {
  engine.load_model("path/to/model.tflite");
  EXPECT_EQ(engine.get_output_channels(), 1000); // Assuming model output channels is 1000
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
  cv::Mat invalid_image(224, 224, CV_8UC1); // Invalid type
  auto result = engine.infer(invalid_image);
  EXPECT_EQ(std::get<0>(result), nullptr);
  EXPECT_EQ(std::get<1>(result), nullptr);
  EXPECT_EQ(std::get<2>(result), nullptr);
  EXPECT_EQ(std::get<3>(result), nullptr);
}

TEST_F(TFLiteInferenceEngineTest, InferReturnsValidTensorsForValidImage) {
  engine.load_model("path/to/model.tflite");
  cv::Mat valid_image(224, 224, CV_8UC3, cv::Scalar(0, 0, 0)); // Valid type
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
  auto status = engine.load_model("path/to/model.tflite");
  EXPECT_EQ(status, tflite::inference::InferenceStatus::OK);
}