find_package(OpenCV)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIR})    
else()
    include(FetchContent)

    set(OpenCV_VERSION 4.8.0)

    FetchContent_Declare(
    OpenCV
    URL "https://github.com/opencv/opencv/archive/${OpenCV_VERSION}.zip"
    )

    FetchContent_MakeAvailable(OpenCV)

    set(BUILD_OPENCV_INCLUDE_DIRS
    ${OPENCV_CONFIG_FILE_INCLUDE_DIR}
    ${OPENCV_MODULE_opencv_core_LOCATION}/include
    ${OPENCV_MODULE_opencv_imgproc_LOCATION}/include
    ${OPENCV_MODULE_opencv_imgcodecs_LOCATION}/include
    PARENT_SCOPE
    )
endif()

set(USED_OPENCV_LIBS opencv_core;opencv_imgproc;opencv_imgcodecs PARENT_SCOPE)
