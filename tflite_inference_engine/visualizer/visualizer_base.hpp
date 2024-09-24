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
#include <log/log.hpp>
#include <opencv2/opencv.hpp>
#include <utils/visualization_status.hpp>
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
  static cv::Mat overlay(const cv::Mat &, const float *, const float *,
                         const float *, const float *, float) {
    return cv::Mat();
  };

public:
  static cv::Mat overlay(const cv::Mat &, const float *, const int &,
                         const int &, const int &) {
    return cv::Mat();
  }

public:
  static inline VisualizationStatus show(const cv::Mat &image,
                                         const std::string &name) {
    if (image.empty()) {
      LOG(ERROR) << "Input image is empty";
      return VisualizationStatus::INPUT_IMAGE_EMPTY;
    }

    if (name.empty()) {
      LOG(ERROR) << "Window name is empty";
      return VisualizationStatus::WINDOW_NAME_EMPTY;
    }

    cv::imshow(name, image);
    cv::waitKey(0);
    return VisualizationStatus::SUCCESS;
  }

public:
  static VisualizationStatus save(const cv::Mat &image,
                                  const std::string &output_path) {
    if (image.empty()) {
      LOG(ERROR) << "Input image is empty";
      return VisualizationStatus::INPUT_IMAGE_EMPTY;
    }

    if (output_path.empty()) {
      LOG(ERROR) << "Output path is empty";
      return VisualizationStatus::OUTPUT_PATH_EMPTY;
    }

    cv::imwrite(output_path, image);
    return VisualizationStatus::SUCCESS;
  }
};
} // namespace tflite::visualizer

#endif // VISUALIZER_BASE_HPP
