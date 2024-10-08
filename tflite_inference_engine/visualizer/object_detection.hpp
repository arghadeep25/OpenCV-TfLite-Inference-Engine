/**
 * @file object_detection.hpp
 * @details Visualizer for object detection
 * @author Arghadeep Mazumder
 * @version 0.1.0
 * @copyright -
 */

#ifndef OBJECT_DETECTION_VISUALIZER_HPP
#define OBJECT_DETECTION_VISUALIZER_HPP


#include <visualizer/visualizer_base.hpp>

namespace tflite::visualizer {
class ObjectDetectionVisualizer : public VisualizerBase {
public:
  ObjectDetectionVisualizer() = default;
  ~ObjectDetectionVisualizer() = default;

  ObjectDetectionVisualizer(const ObjectDetectionVisualizer &) = delete;
  ObjectDetectionVisualizer &
  operator=(const ObjectDetectionVisualizer &) = delete;
  ObjectDetectionVisualizer(ObjectDetectionVisualizer &&) = delete;
  ObjectDetectionVisualizer &operator=(ObjectDetectionVisualizer &&) = delete;

private:
  struct DetectionOutput {
    std::vector<cv::Rect> boxes;
    std::vector<int> classes;
    std::vector<float> scores;
  };

public:
  /**
   * @brief Visualize the detected objects
   * @param image Input image
   * @param boxes Detected boxes
   * @param classes Detected classes
   * @param scores Detected scores
   * @param threshold Detection threshold
   * @return Visualized image
   */
  static cv::Mat overlay(const cv::Mat &image, const float *output_locations,
                         const float *output_classes,
                         const float *output_scores,
                         const float *num_detections, float threshold = 0.5) {
    if (image.empty()) {
      LOG(ERROR) << "Input image is empty";
      return cv::Mat();
    }

    if (output_locations == nullptr || output_classes == nullptr ||
        output_scores == nullptr || num_detections == nullptr) {
      LOG(ERROR) << "Output tensors are nullptr";
      return cv::Mat();
    }

    DetectionOutput output =
        convert_to_array(image.size(), output_locations, output_classes,
                         output_scores, num_detections);

    cv::Mat overlaid_image = image.clone();
    for (size_t i = 0; i < output.boxes.size(); i++) {
      if (output.scores[i] > threshold) {
        cv::rectangle(overlaid_image, output.boxes[i], cv::Scalar(0, 255, 0),
                      2);
        cv::putText(overlaid_image,
                    std::to_string(output.classes[i]) + " : " +
                        std::to_string(output.scores[i]),
                    cv::Point(output.boxes[i].x, output.boxes[i].y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
      }
    }
    return overlaid_image;
  }

private:
  /**
   * @brief Convert the output tensors to array
   * @param size Image size
   * @param output_locations Output locations
   * @param output_classes Output classes
   * @param output_scores Output scores
   * @param num_detections Number of detections
   * @return Detection output
   */
  static DetectionOutput convert_to_array(const cv::Size &size,
                                          const float *output_locations,
                                          const float *output_classes,
                                          const float *output_scores,
                                          const float *num_detections) {
    std::vector<cv::Rect> locations;
    std::vector<int> classes;
    std::vector<float> scores;
    int nums_detected = static_cast<int>(*num_detections);
    DetectionOutput output;

    for (int i = 0; i < nums_detected; ++i) {
      if (output_scores[i] > 0.5) {
        int y1 = static_cast<int>(output_locations[4 * i + 0] * size.width);
        int x1 = static_cast<int>(output_locations[4 * i + 1] * size.height);
        int y2 = static_cast<int>(output_locations[4 * i + 2] * size.width);
        int x2 = static_cast<int>(output_locations[4 * i + 3] * size.height);
        output.boxes.push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
        output.classes.push_back(static_cast<int>(output_classes[i]));
        output.scores.push_back(output_scores[i]);
      }
    }
    return output;
  }
};
} // namespace tflite::visualizer

#endif // OBJECT_DETECTION_VISUALIZER_HPP
