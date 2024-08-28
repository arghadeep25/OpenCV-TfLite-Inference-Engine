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

class ObjectDetectionTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::string modelPath =
        std::string(PROJECT_SOURCE_DIR) + "/models/mobilenet_ssd_v1.tflite";
    object_detection.load_model(modelPath);
  }

  void TearDown() override {}

  TFLiteInferenceEngine object_detection;
};

