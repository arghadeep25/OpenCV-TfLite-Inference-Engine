# OpenCV Tensorflow-Lite Inference Engine

### Overview

#### Object Detection
```cpp

```

#### Segmentation

### Build
```
git clone git@github.com:arghadeep25/OpenCV-TfLite-Inference-Engine.git --recurse-submodules
cd OpenCV-TfLite-Inference-Engine && mkdir build && cd build
cmake ..
make -j$(nproc)
```
### Run
```
cd examples
./object_detection_example
./segmentation_example
```

### Dependencies
- OpenCV
- Tensorflow