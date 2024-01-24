add_executable(StereoMatchApp main.cc)

if (DEFINED BUILD_OPENCV_INCLUDE_DIRS)
    target_include_directories(StereoMatchApp PRIVATE ${BUILD_OPENCV_INCLUDE_DIRS})
endif()

target_link_libraries(StereoMatchApp
    PRIVATE
        ${USED_OPENCV_LIBS}
        StereoMatch
)
