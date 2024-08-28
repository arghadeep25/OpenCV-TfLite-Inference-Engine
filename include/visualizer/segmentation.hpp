//
// Created by arghadeep on 26.08.24.
//

#ifndef SEGMENTATION_VISUALIZER_HPP
#define SEGMENTATION_VISUALIZER_HPP

#include <visualizer/visualizer_base.hpp>

namespace tflite::visualizer {
class SegmentationVisualizer : public VisualizerBase {
public:
  SegmentationVisualizer() = default;
  ~SegmentationVisualizer() = default;

  SegmentationVisualizer(const SegmentationVisualizer &) = delete;
  SegmentationVisualizer &operator=(const SegmentationVisualizer &) = delete;
  SegmentationVisualizer(SegmentationVisualizer &&) = delete;
  SegmentationVisualizer &operator=(SegmentationVisualizer &&) = delete;

public:
  static cv::Mat overlay(const cv::Mat &image, const float *output_locations,
                  const int &height, const int &width, const int &channels) {
    if (image.empty()) {
      LOG_WARNING("Input image is empty");
      return cv::Mat();
    }

    if (output_locations == nullptr) {
      LOG_WARNING("Output locations are nullptr");
      return cv::Mat();
    }

    if (image.size() != cv::Size(width, height)) {
      LOG_WARNING("Input image size does not match with the output size");
      return cv::Mat();
    }

    cv::Mat segmentation_map(height, width, CV_8UC1);
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        float max_prob = 0.0;
        uchar max_class = 0;
        for (int c = 0; c < channels; ++c) {
          float prob =
              output_locations[i * width * channels + j * channels + c];
          if (prob > max_prob) {
            max_prob = prob;
            max_class = c;
          }
        }
        segmentation_map.at<uchar>(i, j) = static_cast<uchar>(max_class * 255.);
      }
    }

    cv::Mat resized_segmentation_map;
    cv::resize(segmentation_map, resized_segmentation_map, image.size(), 0, 0,
               cv::INTER_NEAREST);

    // Apply color map to the segmentation map for visualization
    cv::Mat color_map;
    cv::applyColorMap(resized_segmentation_map, color_map, cv::COLORMAP_JET);

    if (color_map.channels() == 1) {
      cv::cvtColor(color_map, color_map, cv::COLOR_GRAY2BGR);
    }

    if (color_map.type() != image.type()) {
      color_map.convertTo(color_map, image.type());
    }
    cv::Mat output_image;
    cv::addWeighted(image, 0.9, color_map, 0.1, 0, output_image);
    return output_image;
  }
};
} // namespace tflite::visualizer
#endif // SEGMENTATION_VISUALIZER_HPP
