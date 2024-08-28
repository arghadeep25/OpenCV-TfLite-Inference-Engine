/**
 * @file example_segmentation.hpp
 * @details Example script for segmentation
 * @author Arghadeep Mazumder
 * @version 0.1.0
 * @copyright -
 */

#include <infer/infer.hpp>
#include <iostream>
#include <log/log.hpp>
#include <opencv2/opencv.hpp>
#include <utils/scoped_timer.hpp>
#include <visualizer/segmentation.hpp>

int main() {
  std::string img_1_path =
      std::string(PROJECT_SOURCE_DIR) + "/data/person_1.jpg";
  cv::Mat img_1 = cv::imread(img_1_path);
  {
    tflite::inference::TFLiteInferenceEngine segmentation;
    std::string segmentation_model_path =
        std::string(PROJECT_SOURCE_DIR) + "/models/deeplabv3.tflite";
    auto load_model_status = segmentation.load_model(segmentation_model_path);
    if (load_model_status != tflite::inference::InferenceStatus::SUCCESS) {
      LOG_ERROR("Failed to load the model");
      return -1;
    }

    cv::Mat img_1_clone = img_1.clone();
    cv::resize(img_1_clone, img_1_clone,
               cv::Size(segmentation.get_input_height(),
                        segmentation.get_input_width()));
    img_1_clone.convertTo(img_1_clone, CV_32FC3, 1.0 / 255.0);

    auto [output_locations, output_classes, output_scores, num_detections] =
        segmentation.infer(img_1_clone);

    cv::Mat overlayed_image =
        tflite::visualizer::SegmentationVisualizer::overlay(
            img_1_clone, output_locations, segmentation.get_output_height(),
            segmentation.get_output_width(),
            segmentation.get_output_channels());
    auto status = tflite::visualizer::SegmentationVisualizer::show(
        overlayed_image, "Segmentation");
    if (status != tflite::visualizer::VisualizationStatus::SUCCESS) {
      LOG_ERROR("Failed to show the image");
    }
  }
}