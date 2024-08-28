/**
* @file visualizer_base.hpp
* @details Base class for visualizing the output of the model
* @author Arghadeep Mazumder
* @version 0.1.0
* @copyright -
 */

#ifndef VISUALIZER_BASE_HPP
#define VISUALIZER_BASE_HPP

#include <iostream>
#include <logging/log.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace tflite::visualizer {
class VisualizerBase {
public:
  VisualizerBase() = default;
  ~VisualizerBase() = default;

  VisualizerBase(const VisualizerBase &) = delete;
  VisualizerBase &operator=(const VisualizerBase &) = delete;
  VisualizerBase(VisualizerBase &&) = delete;
  VisualizerBase &operator=(VisualizerBase &&) = delete;

public:
  static cv::Mat overlay(const cv::Mat &, const float *, const float *, const float *,
                  const float *, float) {
    return cv::Mat();
  };

public:
  static cv::Mat overlay(const cv::Mat &, const float *, const int &,
                          const int &, const int &) {
    return cv::Mat();
  }

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

public:
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
} // namespace tflite::visualizer

#endif // VISUALIZER_BASE_HPP
