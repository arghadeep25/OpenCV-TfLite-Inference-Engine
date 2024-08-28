/**
* @file log.hpp
* @details Logging utility for the application
* @author Arghadeep Mazumder
* @version 0.1.0
* @copyright -
 */
#ifndef LOGGING_HPP_
#define LOGGING_HPP_

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <utils/colors.hpp>

namespace tflite::logging {
class Logger {
public:
  enum class Level { DEBUG, INFO, WARNING, ERROR, FATAL };

  static Logger &get_instance() {
    static Logger instance;
    return instance;
  }

  template <typename... Args>
  void log(Level level, const char *file, int line, Args... args) {
    std::lock_guard<std::mutex> lock(m_mutex);

    std::stringstream ss;
    ss << get_color(level) << get_level_string(level) << " "
       << get_current_timestamp() << " " << file << ":" << line << " - ";

    (ss << ... << args) << RESET << std::endl;

    m_output << ss.str();
    m_output.flush();
  }

  void setOutput(std::ostream &os) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_output.rdbuf(os.rdbuf());
  }

private:
  Logger() : m_output(std::cout.rdbuf()) {}
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  std::string get_level_string(Level level) const {
    switch (level) {
    case Level::DEBUG:
      return "[DEBUG]";
    case Level::INFO:
      return "[INFO]";
    case Level::WARNING:
      return "[WARNING]";
    case Level::ERROR:
      return "[ERROR]";
    case Level::FATAL:
      return "[FATAL]";
    default:
      return "[UNKNOWN]";
    }
  }

  std::string get_color(Level level) const {
    switch (level) {
    case Level::DEBUG:
      return BLUE;
    case Level::INFO:
      return GREEN;
    case Level::WARNING:
      return YELLOW;
    case Level::ERROR:
      return RED;
    case Level::FATAL:
      return MAGENTA;
    default:
      return RESET;
    }
  }

  std::string get_current_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
  }

  std::mutex m_mutex;
  std::ostream m_output;
};
} // namespace tflite::logging

#define LOG_DEBUG(...)                                                         \
  tflite::logging::Logger::get_instance().log(                                  \
      tflite::logging::Logger::Level::DEBUG, __FILE__, __LINE__, __VA_ARGS__)

#define LOG_INFO(...)                                                          \
  tflite::logging::Logger::get_instance().log(                                  \
      tflite::logging::Logger::Level::INFO, __FILE__, __LINE__, __VA_ARGS__)

#define LOG_WARNING(...)                                                       \
  tflite::logging::Logger::get_instance().log(                                  \
      tflite::logging::Logger::Level::WARNING, __FILE__, __LINE__,             \
      __VA_ARGS__)

#define LOG_ERROR(...)                                                         \
  tflite::logging::Logger::get_instance().log(                                  \
      tflite::logging::Logger::Level::ERROR, __FILE__, __LINE__, __VA_ARGS__)

#define LOG_FATAL(...)                                                         \
  tflite::logging::Logger::get_instance().log(                                  \
      tflite::logging::Logger::Level::FATAL, __FILE__, __LINE__, __VA_ARGS__)

#endif // LOGGING_HPP_