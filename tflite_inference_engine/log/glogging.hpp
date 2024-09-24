//
// Created by arghadeep on 24.09.24.
//

#ifndef TFLITE_INFERENCE_GLOGGING_HPP
#define TFLITE_INFERENCE_GLOGGING_HPP

#include <glog/logging.h>

namespace tflite::logging {
class GLogger {
public:
  static void init(const char *argv0, const std::string &log_dir) {
    google::InitGoogleLogging(argv0);
    google::SetCommandLineOption("log_dir",
                                 log_dir.c_str()); // Set log directory
    std::cout << "Log Dir: " << log_dir << std::endl;
    google::SetCommandLineOption("logtostderr", "0"); // Log to file
    FLAGS_minloglevel = google::GLOG_INFO;            // Set log level
  }

public:
  static void shutdown() {
    google::FlushLogFiles(google::GLOG_INFO);
    google::ShutdownGoogleLogging();
  }
};
} // namespace tflite::logging
#endif // TFLITE_INFERENCE_GLOGGING_HPP
