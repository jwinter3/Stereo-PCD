include(FetchContent)

set(CATCH_VERSION 3.0.1)

FetchContent_Declare(
    Catch2
    URL "https://github.com/catchorg/Catch2/archive/v${CATCH_VERSION}.zip"
)

FetchContent_MakeAvailable(Catch2)

add_executable(StereoMatchTests StereoMatchTests.cc)

if (DEFINED BUILD_OPENCV_INCLUDE_DIRS)
    target_include_directories(StereoMatchTests PRIVATE ${BUILD_OPENCV_INCLUDE_DIRS})
endif()

target_link_libraries(StereoMatchTests StereoMatch ${USED_OPENCV_LIBS} Catch2::Catch2WithMain)
add_test(NAME StereoMatchTests COMMAND StereoMatchTests)
