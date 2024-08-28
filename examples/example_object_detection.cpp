/**
* @file example_object_detection.hpp
* @details Example script for object detection
* @author Arghadeep Mazumder
* @version 0.1.0
* @copyright -
*/

#include <infer/infer.hpp>
#include <iostream>
#include <log/log.hpp>
#include <opencv2/opencv.hpp>
#include <utils/scoped_timer.hpp>
#include <visualizer/object_detection.hpp>

int main() {
  std::string img_1_path =
      std::string(PROJECT_SOURCE_DIR) + "/data/person_1.jpg";
  cv::Mat img_1 = cv::imread(img_1_path);

  // Object Detection
  {
    std::string modelPath =
        std::string(PROJECT_SOURCE_DIR) + "/models/mobilenet_ssd_v1.tflite";
    tflite::inference::TFLiteInferenceEngine object_detection;

    object_detection.load_model(modelPath);
    cv::Mat img_1_clone = img_1.clone();
    cv::resize(img_1_clone, img_1_clone,
               cv::Size(object_detection.get_input_height(),
                        object_detection.get_input_width()));

    auto [output_locations, output_classes, output_scores, num_detections] =
        object_detection.infer(img_1_clone);

    cv::Mat overlayed_image =
        tflite::visualizer::ObjectDetectionVisualizer::overlay(
            img_1_clone, output_locations, output_classes, output_scores,
            num_detections);

    auto status = tflite::visualizer::ObjectDetectionVisualizer::show(
        overlayed_image, "");
    if (status != tflite::visualizer::VisualizationStatus::SUCCESS) {
      LOG_ERROR("Failed to show the image");
    }
  }
}