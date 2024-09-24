/**
 * @file example_object_detection.hpp
 * @details Example script for object detection
 * @author Arghadeep Mazumder
 * @version 0.1.0
 * @copyright -
 */

#include <infer/infer.hpp>
#include <iostream>
#include <log/glogging.hpp>
#include <log/log.hpp>
#include <opencv2/opencv.hpp>
#include <utils/scoped_timer.hpp>
#include <visualizer/object_detection.hpp>

int main(int argc, char **argv) {
  tflite::logging::GLogger::init(argv[0],
                                 std::string(PROJECT_SOURCE_DIR) + "/logs");
  std::string img_1_path =
      std::string(PROJECT_SOURCE_DIR) + "/data/person_3.jpg";
  cv::Mat img_1 = cv::imread(img_1_path);
  if (img_1.empty()) {
    LOG(ERROR) << "Failed to read the image";
    tflite::logging::GLogger::shutdown();
    return -1;
  } else {
    LOG(INFO) << "Image Loaded Successfully";
  }

  // Object Detection
  {
    LOG(INFO) << "Object Detection Started";
    std::string modelPath =
        std::string(PROJECT_SOURCE_DIR) + "/models/mobilenet_ssd_v12.tflite";
    tflite::inference::TFLiteInferenceEngine object_detection;

    if (object_detection.load_model(modelPath) ==
        tflite::inference::InferenceStatus::SUCCESS) {
      LOG(INFO) << "Model Loaded Successfully";
    } else {
      LOG(ERROR) << "Failed to load the model";
      tflite::logging::GLogger::shutdown();
      return -1;
    }
    cv::Mat img_1_clone = img_1.clone();
    cv::resize(img_1_clone, img_1_clone,
               cv::Size(object_detection.get_input_height(),
                        object_detection.get_input_width()));

    auto [output_locations, output_classes, output_scores, num_detections] =
        object_detection.infer(img_1_clone);
    LOG(INFO) << "Inference Completed Successfully";

    cv::Mat overlayed_image =
        tflite::visualizer::ObjectDetectionVisualizer::overlay(
            img_1_clone, output_locations, output_classes, output_scores,
            num_detections);

    auto status = tflite::visualizer::ObjectDetectionVisualizer::show(
        overlayed_image, "Detection");
    if (status != tflite::visualizer::VisualizationStatus::SUCCESS) {
      LOG(ERROR) << "Failed to show the image";
      tflite::logging::GLogger::shutdown();
      return -1;
    }
  }
  tflite::logging::GLogger::shutdown();
  return 0;
}