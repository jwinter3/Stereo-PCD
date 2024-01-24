cmake_minimum_required(VERSION 3.1.4)
project(StereoMatch VERSION 0.1.0)

set (CMAKE_C_STANDARD 11)
set (CMAKE_CXX_STANDARD 17)

add_compile_options(-O3 -fPIC)

add_subdirectory(dependencies)

if (UNIX)
add_compile_options(-Wall -Wextra -pedantic -Werror -Wno-c11-extensions -O3 -fPIC)
elseif(MSVC)
add_compile_options(-W3 -EHsc -WX)
endif()

include_directories(lib)

enable_testing()

add_subdirectory(lib)
add_subdirectory(module)
add_subdirectory(tests)
add_subdirectory(app)

set_target_properties(StereoMatchApp StereoMatch StereoMatchTests stereo_match
    PROPERTIES
        CXX_STANDARD_REQUIRED ON
        CXX_CLANG_TIDY clang-tidy
)

set_target_properties(StereoMatchApp StereoMatch
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/app"
)

set_target_properties(StereoMatchTests
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
)
