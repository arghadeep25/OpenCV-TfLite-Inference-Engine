
#include <inference_engine/inference_engine.hpp>
#include <iostream>
#include <logging/log.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <utils/scoped_timer.hpp>
#include <visualizer/object_detection.hpp>
#include <visualizer/segmentation.hpp>

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

    tflite::visualizer::ObjectDetectionVisualizer visualizer;
    cv::Mat overlayed_image =
        visualizer.overlay(
            img_1_clone, output_locations, output_classes, output_scores,
            num_detections);
    tflite::visualizer::ObjectDetectionVisualizer::show(overlayed_image,
                                                        "Object Detection");
  }


  // Segmentation
  {
    tflite::inference::TFLiteInferenceEngine segmentation;
    std::string segmentation_model_path =
        std::string(PROJECT_SOURCE_DIR) + "/models/deeplabv3.tflite";
    segmentation.load_model(segmentation_model_path);

    cv::Mat img_1_clone = img_1.clone();
    cv::resize(img_1_clone, img_1_clone,
               cv::Size(segmentation.get_input_height(),
                        segmentation.get_input_width()));
    img_1_clone.convertTo(img_1_clone, CV_32FC3, 1.0 / 255.0);

    auto [output_locations, output_classes, output_scores, num_detections] =
        segmentation.infer(img_1_clone);

    cv::Mat overlayed_image = tflite::visualizer::SegmentationVisualizer::overlay(
        img_1_clone, output_locations, segmentation.get_output_height(),
        segmentation.get_output_width(), segmentation.get_output_channels());
    tflite::visualizer::SegmentationVisualizer::show(overlayed_image,
                                                     "Segmentation");
  }
}