add_library(StereoMatch StereoMatch/StereoMatch.cc
            StereoMatch/StereoMatchSingleThread.cc
            StereoMatch/StereoMatchMultiThreads.cc
            )

if (DEFINED BUILD_OPENCV_INCLUDE_DIRS)
    target_include_directories(StereoMatch PRIVATE ${BUILD_OPENCV_INCLUDE_DIRS})
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(StereoMatch
    PRIVATE
        ${USED_OPENCV_LIBS}
        Threads::Threads
)
