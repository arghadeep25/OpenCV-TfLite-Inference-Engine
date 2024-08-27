//
// Created by arghadeep on 27.08.24.
//

#ifndef VISUALIZER_BASE_HPP
#define VISUALIZER_BASE_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <logging/log.hpp>
#include <vector>


namespace tflite::visualizer {
class VisualizerBase{
public:
  VisualizerBase() = default;
  ~VisualizerBase() = default;

  VisualizerBase(const VisualizerBase &) = delete;
  VisualizerBase &operator=(const VisualizerBase &) = delete;
  VisualizerBase(VisualizerBase &&) = delete;
  VisualizerBase &operator=(VisualizerBase &&) = delete;

 public:
  virtual cv::Mat
  overlay(const cv::Mat &image, const float *output_locations,
          const float *output_classes, const float *output_scores,
          const float *num_detections, float threshold) = 0;

 public:
  virtual cv::Mat
  overlay(const cv::Mat &image, const float *output_locations,
          const int &height, const int &width, const int &channels) = 0;

 public:
  static inline void show(const cv::Mat &image, const std::string &name) {
    if (image.empty()) {
      LOG_WARNING("Input image is empty");
      return;
    }

    if (name.empty()) {
      LOG_WARNING("Window name is empty");
      return;
    }

    cv::imshow(name, image);
    cv::waitKey(0);
  }

  static void save(const cv::Mat &image, const std::string &output_path) {
    if (image.empty()) {
      LOG_WARNING("Input image is empty");
      return;
    }

    if (output_path.empty()) {
      LOG_WARNING("Output path is empty");
      return;
    }

    cv::imwrite(output_path, image);
  }
};
}


#endif //VISUALIZER_BASE_HPP
