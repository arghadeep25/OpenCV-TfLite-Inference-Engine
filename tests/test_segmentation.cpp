/**
 * @file test_segmentation.hpp
 * @details Test cases for segmentation
 * @author Arghadeep Mazumder
 * @version 0.1.0
 * @copyright -
 */

#include <gtest/gtest.h>
#include <infer/infer.hpp>
#include <log/log.hpp>
#include <opencv2/opencv.hpp>
#include <utils/scoped_timer.hpp>
#include <visualizer/segmentation.hpp>

using namespace tflite::inference;
using namespace tflite::visualizer;

class SegmentationTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::string model_path =
        std::string(PROJECT_SOURCE_DIR) + "/models/deeplabv3.tflite";
    auto status = segmentation.load_model(model_path);
    assert(stauts == InferenceStatus::SUCCESS);
  }

  void TearDown() override {}
  TFLiteInferenceEngine segmentation;

  // Input
  const int input_height = 257;
  const int input_width = 257;
  const int input_channels = 3;
  // Output
  const int output_height = 257;
  const int output_width = 257;
  const int output_channels = 21;

  // Input Image Path
  std::string image_path =
      std::string(PROJECT_SOURCE_DIR) + "/data/person_1.jpg";
};

TEST_F(SegmentationTest, GetInputHeightReturnsCorrectValue) {
  EXPECT_EQ(segmentation.get_input_height(), this->input_height);
}

TEST_F(SegmentationTest, GetInputHeightReturnsWrongValue) {
  EXPECT_NE(segmentation.get_input_height(), 0);
}

TEST_F(SegmentationTest, GetInputWidthReturnsCorrectValue) {
  EXPECT_EQ(segmentation.get_input_width(), this->input_width);
}

TEST_F(SegmentationTest, GetInputWidthReturnsWrongValue) {
  EXPECT_NE(segmentation.get_input_width(), 0);
}

TEST_F(SegmentationTest, GetInputChannelsReturnsCorrectValue) {
  EXPECT_EQ(segmentation.get_input_channels(), this->input_channels);
}

TEST_F(SegmentationTest, GetInputChannelsReturnsWrongValue) {
  EXPECT_NE(segmentation.get_input_channels(), 0);
}

TEST_F(SegmentationTest, GetOutputHeightReturnsCorrectValue) {
  EXPECT_EQ(segmentation.get_output_height(), this->output_height);
}

TEST_F(SegmentationTest, GetOutputHeightReturnsWrongValue) {
  EXPECT_NE(segmentation.get_output_height(), 0);
}

TEST_F(SegmentationTest, GetOutputWidthReturnsCorrectValue) {
  EXPECT_EQ(segmentation.get_output_width(), this->output_width);
}

TEST_F(SegmentationTest, GetOutputWidthReturnsWrongValue) {
  EXPECT_NE(segmentation.get_output_width(), 0);
}

TEST_F(SegmentationTest, GetOutputChannelsReturnsCorrectValue) {
  EXPECT_EQ(segmentation.get_output_channels(), this->output_channels);
}

TEST_F(SegmentationTest, GetOutputChannelsReturnsWrongValue) {
  EXPECT_NE(segmentation.get_output_channels(), 0);
}

TEST_F(SegmentationTest, InferSegmentationReturnsCorrectOutput) {
  cv::Mat image = cv ::imread(this->image_path);
  assert(!image.empty());
  cv::resize(image, image,
             cv::Size(segmentation.get_input_height(),
                      segmentation.get_input_width()));
  image.convertTo(image, CV_32FC3, 1.0 / 255.0);
  auto [output_locations, output_classes, output_scores, num_detections] =
      segmentation.infer(image);

  EXPECT_NE(output_locations, nullptr);
}

TEST_F(SegmentationTest, InferSegmentationWONormalizationReturnsCorrectOutput) {
  cv::Mat image = cv ::imread(this->image_path);
  assert(!image.empty());
  cv::resize(image, image,
             cv::Size(segmentation.get_input_height(),
                      segmentation.get_input_width()));
  auto [output_locations, output_classes, output_scores, num_detections] =
      segmentation.infer(image);
  EXPECT_NE(output_locations, nullptr);
}

TEST_F(SegmentationTest, InferSegmentationReturnsNullptrForEmptyImage) {
  cv::Mat empty_image;
  auto result = segmentation.infer(empty_image);
  EXPECT_EQ(std::get<0>(result), nullptr);
  EXPECT_EQ(std::get<1>(result), nullptr);
  EXPECT_EQ(std::get<2>(result), nullptr);
  EXPECT_EQ(std::get<3>(result), nullptr);
}


