/**
 * @file test_object_detection.hpp
 * @details Test cases for object detection
 * @author Arghadeep Mazumder
 * @version 0.1.0
 * @copyright -
 */

#include <gtest/gtest.h>
#include <infer/infer.hpp>
#include <log/log.hpp>
#include <opencv2/opencv.hpp>
#include <utils/scoped_timer.hpp>
#include <visualizer/object_detection.hpp>

using namespace tflite::inference;
using namespace tflite::visualizer;

class ObjectDetectionTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::string model_path =
        std::string(PROJECT_SOURCE_DIR) + "/models/mobilenet_ssd_v1.tflite";
    object_detection.load_model(model_path);
  }

  void TearDown() override {}

  TFLiteInferenceEngine object_detection;

  // Input
  const int input_height = 300;
  const int input_width = 300;
  const int input_channels = 3;
  // Output
  const int output_height = 10;
  const int output_width = 4;
  const int output_channels = 0;

  // Input Image Path
  std::string image_path =
      std::string(PROJECT_SOURCE_DIR) + "/data/person_1.jpg";
};

TEST_F(ObjectDetectionTest, GetInputHeightReturnsCorrectValue) {
  EXPECT_EQ(object_detection.get_input_height(), this->input_height);
}

TEST_F(ObjectDetectionTest, GetInputHeightReturnsWrongValue) {
  EXPECT_NE(object_detection.get_input_height(), 0);
}

TEST_F(ObjectDetectionTest, GetInputWidthReturnsCorrectValue) {
  EXPECT_EQ(object_detection.get_input_width(), this->input_width);
}

TEST_F(ObjectDetectionTest, GetInputWidthReturnsWrongValue) {
  EXPECT_NE(object_detection.get_input_width(), 0);
}

TEST_F(ObjectDetectionTest, GetInputChannelsReturnsCorrectValue) {
  EXPECT_EQ(object_detection.get_input_channels(), this->input_channels);
}

TEST_F(ObjectDetectionTest, GetInputChannelsReturnsWrongValue) {
  EXPECT_NE(object_detection.get_input_channels(), 1000);
}

TEST_F(ObjectDetectionTest, GetOutputHeightReturnsCorrectValue) {
  EXPECT_EQ(object_detection.get_output_height(), this->output_height);
}

TEST_F(ObjectDetectionTest, GetOutputHeightReturnsWrongValue) {
  EXPECT_NE(object_detection.get_output_height(), 0);
}

TEST_F(ObjectDetectionTest, GetOutputWidthReturnsCorrectValue) {
  EXPECT_EQ(object_detection.get_output_width(), this->output_width);
}

TEST_F(ObjectDetectionTest, GetOutputWidthReturnsWrongValue) {
  EXPECT_NE(object_detection.get_output_width(), 0);
}

TEST_F(ObjectDetectionTest, GetOutputChannelsReturnsWrongValue) {
  EXPECT_NE(object_detection.get_output_channels(), 1000);
}

TEST_F(ObjectDetectionTest, InferObjectDetection) {
  cv::Mat image = cv::imread(this->image_path);
  cv::resize(image, image,
             cv::Size(object_detection.get_input_height(),
                      object_detection.get_input_width()));

  auto [output_locations, output_classes, output_scores, num_detections] =
      object_detection.infer(image);

  EXPECT_NE(output_locations, nullptr);
  EXPECT_NE(output_classes, nullptr);
  EXPECT_NE(output_scores, nullptr);
  EXPECT_NE(num_detections, nullptr);
}

TEST_F(ObjectDetectionTest, InferReturnsNullptrsForEmptyImage) {
  cv::Mat empty_image;
  auto result = object_detection.infer(empty_image);
  EXPECT_EQ(std::get<0>(result), nullptr);
  EXPECT_EQ(std::get<1>(result), nullptr);
  EXPECT_EQ(std::get<2>(result), nullptr);
  EXPECT_EQ(std::get<3>(result), nullptr);
}

TEST_F(ObjectDetectionTest,
       InferReturnsValidImageVisReturnsValidOverlayedImage) {
  cv::Mat image = cv::imread(this->image_path);
  cv::resize(image, image,
             cv::Size(this->object_detection.get_input_height(),
                      this->object_detection.get_input_width()));

  EXPECT_EQ(image.size().height, this->object_detection.get_input_height());
  EXPECT_EQ(image.size().width, this->object_detection.get_input_width());

  auto [output_locations, output_classes, output_scores, num_detections] =
      this->object_detection.infer(image);
  cv::Mat result = tflite::visualizer::ObjectDetectionVisualizer::overlay(
      image, output_locations, output_classes, output_scores, num_detections);

  EXPECT_EQ(result.empty(), false);
}

TEST_F(ObjectDetectionTest,
       InferReturnsValidZeroImageVisReturnsValidOverlayedImage) {
  cv::Mat image =
      cv::Mat::zeros(this->object_detection.get_input_height(),
                     this->object_detection.get_input_width(), CV_8UC3);

  auto [output_locations, output_classes, output_scores, num_detections] =
      this->object_detection.infer(image);
  cv::Mat result = tflite::visualizer::ObjectDetectionVisualizer::overlay(
      image, output_locations, output_classes, output_scores, num_detections);

  EXPECT_EQ(result.empty(), false);
}

TEST_F(ObjectDetectionTest, OverlayTesting) {
  cv::Mat image;
  cv::Mat result = tflite::visualizer::ObjectDetectionVisualizer::overlay(
      image, nullptr, nullptr, nullptr, nullptr);
  EXPECT_EQ(result.empty(), true);
}
