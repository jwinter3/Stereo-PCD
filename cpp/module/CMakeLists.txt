find_package(PythonLibs 3.8 REQUIRED)
find_package(Boost COMPONENTS python REQUIRED)
find_package(Boost COMPONENTS numpy REQUIRED)

set(CMAKE_SHARED_MODULE_PREFIX "")


add_library(stereo_match MODULE StereoMatchModule.cc DataTypes.cc PythonStereoMatch.cc)

# Set up the libraries and header search paths for this target
target_link_libraries(stereo_match ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${USED_OPENCV_LIBS} StereoMatch)
target_include_directories(stereo_match PRIVATE ${PYTHON_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_custom_command(
    TARGET stereo_match
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/stereo_match.so ${CMAKE_CURRENT_SOURCE_DIR}/../../stereo_pcd/stereo_match.so
    COMMENT "Copying stereo_match.so to stereo_pcd..."
)
