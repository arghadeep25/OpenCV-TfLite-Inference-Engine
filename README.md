# OpenCV Tensorflow-Lite Inference Engine

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/3c62ad3569e94f5a8063b335538cb7f4)](https://app.codacy.com/gh/arghadeep25/OpenCV-TfLite-Inference-Engine/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)

### Overview

This repository provides a simple C++ API to perform inference using Tensorflow-Lite models with OpenCV. The API is designed to be simple and easy to use. The API is designed to be modular and can be easily extended to support other models.

#### Object Detection

```cpp
// Load the inference header
#include <infer/infer.hpp>
// Load visualization header
#include <visualizer/object_detection.hpp>

// Define the object
tflite::inference::TFLiteInferenceEngine object_detection;

// Load the model
object_detection.load_model(model_path);

// Resize the image
cv::resize(image, image,
           cv::Size(object_detection.get_input_height(),
                    object_detection.get_input_width()));

// Perform inference
auto [output_locations, output_classes, output_scores, num_detections] =
    object_detection.infer(image);

// Visualize the output
cv::Mat output =
        tflite::visualizer::ObjectDetectionVisualizer::overlay(
            image, output_locations, output_classes, output_scores,
            num_detections);
```

#### Segmentation
```cpp
// Load the inference header
#include <infer/infer.hpp>
// Load visualization header
#include <visualizer/segmentation.hpp>

// Define the object
tflite::inference::TFLiteInferenceEngine segmentation;

// Load the model
segmentation.load_model(model_path);

// Resize and Normalize the image
cv::resize(image, image,
           cv::Size(segmentation.get_input_height(),
                    segmentation.get_input_width()));
image.convertTo(image, CV_32FC3, 1.0 / 255.0);

// Perform inference
auto [output_locations, output_classes, output_scores, num_detections] =
    segmentation.infer(image);

// Visualize the output
cv::Mat output = tflite::visualizer::SegmentationVisualizer::overlay(
    image, output_locations, segmentation.get_output_height(),
            segmentation.get_output_width(),
            segmentation.get_output_channels());

```

### Build

```
git clone git@github.com:arghadeep25/OpenCV-TfLite-Inference-Engine.git --recurse-submodules
cd OpenCV-TfLite-Inference-Engine 
cmake -S . -B build && cmake --build build -j$(nproc)
```

### Run Examples

```
./build/examples/object_detection_example
./build/examples/segmentation_example
```

### Run Tests

```
./build/tests/runTests
```

### Dependencies

- OpenCV
- Tensorflow
- glog
- gflags
- C++ 17

### Logs Viewer

Open a terminal and run the following command to open the GUI
```
glogg
```
and then open the log files from the logs directory..