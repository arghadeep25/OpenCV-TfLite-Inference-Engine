#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <inference_engine/inference_engine.hpp>
#include <utils/scoped_timer.hpp>


void drawBoundingBoxes(cv::Mat &img, const std::vector<cv::Rect> &boxes,
                       const std::vector<int> &class_ids,
                       const std::vector<float> &scores) {
  for (size_t i = 0; i < boxes.size(); ++i) {
    cv::rectangle(img, boxes[i], cv::Scalar(0, 255, 0),
                  2);  // Green bounding box
    std::string label = "Class: " + std::to_string(class_ids[i]) + " (" +
        std::to_string(int(scores[i] * 100)) + "%)";
    int baseLine;
    cv::Size labelSize =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    cv::rectangle(
        img, cv::Point(boxes[i].x, boxes[i].y - labelSize.height),
        cv::Point(boxes[i].x + labelSize.width, boxes[i].y + baseLine),
        cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(img, label, cv::Point(boxes[i].x, boxes[i].y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
  }
}

int main() {
  cv::Mat img = cv::Mat::zeros(300, 300, CV_8UC3);
  std::cout << "Img size: " << img.size() << std::endl;

  std::string modelPath =
      std::string(PROJECT_SOURCE_DIR) + "/models/mobilenet_ssd_v1.tflite";
  std::cout << modelPath << std::endl;

  tflite::inference::TFLiteInferenceEngine object_detection;
  std::cout << "Loading model" << std::endl;
  object_detection.load_model(modelPath);
//  {
//    utils::timer::ScopedTimer timer;
    auto res = object_detection.infer(img);
    // access tuple elements
    auto output_locations = std::get<0>(res);
    auto output_classes = std::get<1>(res);
    auto output_scores = std::get<2>(res);
    auto num_detections = std::get<3>(res);
    int numDetections = static_cast<int>(num_detections->data.f[0]);
    auto output_data = output_locations->data.f;
    for (int i = 0; i < output_locations->bytes; i++) {
      std::cout << output_data[i] << std::endl;
    }
//  }

  std::string pose_model_path =
      std::string(PROJECT_SOURCE_DIR) + "/models/movenet.tflite";
  tflite::inference::TFLiteInferenceEngine pose_estimation;
  pose_estimation.load_model(pose_model_path);
  {
    utils::timer::ScopedTimer timer;
    pose_estimation.infer(img);
  }


  tflite::inference::TFLiteInferenceEngine segmentation;
  std::string segmentation_model_path =
      std::string(PROJECT_SOURCE_DIR) + "/models/deeplabv3.tflite";
  segmentation.load_model(segmentation_model_path);
  cv::Mat img_2 = cv::Mat::zeros(segmentation.get_input_height(),
                                 segmentation.get_input_width(), CV_8UC3);
  {
    utils::timer::ScopedTimer timer;
    segmentation.infer(img_2);
  }


//  std::string image_path = std::string(PROJECT_SOURCE_DIR) + "/data/test.jpg";
//  cv::Mat img = cv::imread(image_path);
//  //   cv::Mat img = cv::Mat::zeros(300, 300, CV_8UC3);
//  cv::resize(img, img, cv::Size(300, 300));
//
//  std::string modelPath =
//      std::string(PROJECT_SOURCE_DIR) + "/models/mobilenet_ssd_v1.tflite";
//  std::cout << modelPath << std::endl;
//  // Load model
//  auto model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
//  // if (!model) {
//  //   std::cerr << "Failed to load model" << std::endl;
//  //   return nullptr;
//  // }
//
//  // Create the interpreter
//  tflite::ops::builtin::BuiltinOpResolver resolver;
//  std::unique_ptr<tflite::Interpreter> interpreter;
//  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
//  // auto interpreter = load_model(modelPath);
//  if (!interpreter) throw std::runtime_error("Failed to load model");
//
//  // Interpreter expectations
//  TfLiteIntArray* dims = interpreter->tensor(interpreter->inputs()[0])->dims;
//  int inputHeight = dims->data[1];
//  int inputWidth = dims->data[2];
//  int inputChannels = dims->data[3];
//  std::cout << "Interpreter expectations" << std::endl;
//  std::cout << "Input Height: " << inputHeight << std::endl;
//  std::cout << "Input Width: " << inputWidth << std::endl;
//  std::cout << "Input Channels: " << inputChannels << std::endl;
//
//  interpreter->AllocateTensors();
//
//  int input = interpreter->inputs()[0];
//  auto input_tensor = interpreter->tensor(input);
//
//  memcpy(interpreter->typed_input_tensor<uchar>(0), img.data,
//         img.total() * img.elemSize());
//  // interpreter->SetAllowFp16PrecisionForFp32(true);
//
//  interpreter->SetNumThreads(4);
//  interpreter->Invoke();
//
//  auto output_locations = interpreter->tensor(interpreter->outputs()[0]);
//  auto output_data = output_locations->data.f;
//
//  // for (int i = 0; i < output_locations->bytes; i++) {
//  //   std::cout << output_data[i] << std::endl;
//  // }
//
//  // std::vector<float> locations;
//
//  auto output_classes = interpreter->tensor(interpreter->outputs()[1]);
//  auto output_data_classes = output_classes->data.f;
//  for (int i = 0; i < output_classes->bytes; i++) {
//    std::cout << output_data_classes[i] << " ";
//  }
//  std::cout << std::endl;
//
//  auto output_scores = interpreter->tensor(interpreter->outputs()[2]);
//  auto output_data_scores = output_scores->data.f;
//  for (int i = 0; i < output_scores->bytes; i++) {
//    std::cout << output_data_scores[i] << " ";
//  }
//  std::cout << std::endl;
//
//  auto num_detections = interpreter->tensor(interpreter->outputs()[3]);
//  auto num_detections_data = num_detections->data.f;
//  std::cout << "Number of detections: ";
//  for (int i = 0; i < num_detections->bytes; i++) {
//    std::cout << num_detections_data[i] << " ";
//  }
//  std::cout << std::endl;
//
//  // auto detections = num_detections->data.f;
//  // std::cout << "Size of detections: " << detections.size() << std::endl;
//
//  std::cout << "Output Locations: " << output_locations << std::endl;
//  std::cout << "Output Classes: " << output_classes << std::endl;
//  std::cout << "Output Scores: " << output_scores << std::endl;
//  // std::cout << "Number of detections: " << num_detections->data << std::endl;
//
//  // interpreter->SeAllow
//
//  // interpreter->AllocateTensors();
//
//  // uchar* input = interpreter->typed_input_tensor<uchar>(0);
//
//  // auto image_height
//
//  float* outputLocations =
//      interpreter->typed_output_tensor<float>(0);  // bounding boxes
//  float* outputClasses =
//      interpreter->typed_output_tensor<float>(1);  // class ids
//  float* outputScores =
//      interpreter->typed_output_tensor<float>(2);  // confidence scores
//  float* numDetections =
//      interpreter->typed_output_tensor<float>(3);  // number of detections
//
//  if (!outputLocations || !outputClasses || !outputScores || !numDetections) {
//    std::cerr << "Failed to obtain output tensors" << std::endl;
//    return -1;
//  }
//
//  std::cout << "Output Locations: " << outputLocations << std::endl;
//
//  std::cout << "Number of detections: " << numDetections[0] << std::endl;
//
  // Post-process the results
  int numDetected = static_cast<int>(numDetections);
  std::vector<cv::Rect> boxes;
  std::vector<int> class_ids;
  std::vector<float> scores;

  for (int i = 0; i < numDetected; ++i) {
    if (outputScores[i] > 0.8) {  // threshold
      int y1 = static_cast<int>(outputLocations[4 * i + 0] * img.rows);
      int x1 = static_cast<int>(outputLocations[4 * i + 1] * img.cols);
      int y2 = static_cast<int>(outputLocations[4 * i + 2] * img.rows);
      int x2 = static_cast<int>(outputLocations[4 * i + 3] * img.cols);
      boxes.push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
      class_ids.push_back(static_cast<int>(outputClasses[i]));
      scores.push_back(outputScores[i]);
    }
  }

  // Draw bounding boxes on the image
  drawBoundingBoxes(img, boxes, class_ids, scores);

  // Display the image with bounding boxes
  cv::imshow("Object Detection", img);
  cv::waitKey(0);  // Wait for a key press
//
//  // Save the result
//  cv::imwrite("result.jpg", img);

  //   return 0;
}