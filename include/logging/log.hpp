#ifndef LOGGING_HPP_
#define LOGGING_HPP_

#include <iostream>

template <typename T> void LOG(T msg) { std::cout << msg << std::endl; }

template <typename T, typename... Args> void LOG(T msg, Args... args) {
  std::cout << msg;
  LOG(args...);
}

#endif // LOGGING_HPP_
