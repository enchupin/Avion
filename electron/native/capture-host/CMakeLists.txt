cmake_minimum_required(VERSION 3.22)

project(AvionCaptureHost LANGUAGES CXX)

add_executable(AvionCaptureHost WIN32
  src/capture.cpp
)

target_compile_features(AvionCaptureHost PRIVATE cxx_std_20)

# Keep the packaged capture host runnable without a separate MSVC runtime install.
set_property(TARGET AvionCaptureHost PROPERTY
  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

target_compile_definitions(AvionCaptureHost PRIVATE
  NOMINMAX
  UNICODE
  _UNICODE
  WIN32_LEAN_AND_MEAN
)

target_link_libraries(AvionCaptureHost PRIVATE
  d3d11
  dxgi
  shell32
  user32
  windowscodecs
  windowsapp
)
