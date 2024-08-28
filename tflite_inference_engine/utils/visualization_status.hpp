//
// Created by arghadeep on 28.08.24.
//

#ifndef VISUALIZATION_STATUS_HPP
#define VISUALIZATION_STATUS_HPP

namespace tflite::visualizer {
enum class VisualizationStatus {
  SUCCESS,
  INPUT_IMAGE_EMPTY,
  WINDOW_NAME_EMPTY,
  OUTPUT_PATH_EMPTY
};
} // namespace tflite::visualizer

#endif // VISUALIZATION_STATUS_HPP
