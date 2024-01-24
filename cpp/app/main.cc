#include "StereoMatch/StereoMatchMultiThreads.h"
#include "StereoMatch/StereoMatchSingleThread.h"

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <memory>

int main(int argc, char *argv[]) {
    float start_gap_cost = -0.1;
    float continue_gap_cost = -0.01;
    float match_reward = 0.1;
    float corner_ths = 0.0001;
    float output_multiplier = 128;
    int max_disp = 0;
    int num_threads = 4;

    if (argc < 4 || argc > 11) {
        std::cerr << "Usage: path/to/left_img path/to/right_img path/to/output "
                     "[start_gap_cost] [continue_gap_cost] [match_reward] "
                     "[corner_ths] [max_disp] [num_threads] [output_multiplier]"
                  << std::endl;
        exit(1);
    }

    if (argc >= 5) {
        try {
            start_gap_cost = std::stof(argv[4]);
        } catch (const std::invalid_argument &e) {
            std::cerr << argv[4] << " is not valid float" << std::endl;
            exit(2);
        }
    }

    if (argc >= 6) {
        try {
            continue_gap_cost = std::stof(argv[5]);
        } catch (const std::invalid_argument &e) {
            std::cerr << argv[5] << " is not valid float" << std::endl;
            exit(2);
        }
    }

    if (argc >= 7) {
        try {
            match_reward = std::stof(argv[6]);
        } catch (const std::invalid_argument &e) {
            std::cerr << argv[6] << " is not valid float" << std::endl;
            exit(2);
        }
    }

    if (argc >= 8) {
        try {
            corner_ths = std::stof(argv[7]);
        } catch (const std::invalid_argument &e) {
            std::cerr << argv[7] << " is not valid float" << std::endl;
            exit(2);
        }
    }

    if (argc >= 9) {
        try {
            max_disp = std::stoi(argv[8]);
        } catch (const std::invalid_argument &e) {
            std::cerr << argv[8] << " is not valid int" << std::endl;
            exit(2);
        }
    }

    if (argc >= 10) {
        try {
            num_threads = std::stoi(argv[9]);
        } catch (const std::invalid_argument &e) {
            std::cerr << argv[9] << " is not valid int" << std::endl;
            exit(2);
        }
    }

    if (argc >= 11) {
        try {
            output_multiplier = std::stoi(argv[10]);
        } catch (const std::invalid_argument &e) {
            std::cerr << argv[10] << " is not valid int" << std::endl;
            exit(2);
        }
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);

    if (image.dims == 0) {
        std::cerr << "Cannot read file " << argv[1] << std::endl;
        exit(3);
    }

    cv::Mat image1 = cv::imread(argv[2], cv::IMREAD_COLOR);

    if (image.dims == 0) {
        std::cerr << "Cannot read file " << argv[2] << std::endl;
        exit(3);
    }

    cv::Mat image_yuv(image.rows, image.cols, CV_8UC3);
    cv::Mat image1_yuv(image1.rows, image1.cols, CV_8UC3);

    cv::Mat image_intensity(image.rows, image.cols, CV_8UC1);
    cv::Mat image1_intensity(image1.rows, image1.cols, CV_8UC1);

    cv::Mat image_harris(image.rows, image.cols, CV_32FC1);
    cv::Mat image1_harris(image1.rows, image1.cols, CV_32FC1);

    cv::cvtColor(image, image_yuv, cv::COLOR_BGR2YUV);
    cv::cvtColor(image1, image1_yuv, cv::COLOR_BGR2YUV);

    cv::extractChannel(image_yuv, image_intensity, 0);
    cv::extractChannel(image1_yuv, image1_intensity, 0);

    cv::cornerHarris(image_intensity, image_harris, 2, 3, 0.04);
    cv::cornerHarris(image1_intensity, image1_harris, 2, 3, 0.04);

    // // for debug
    // cv::Mat edges(image.rows, image.cols, CV_8UC1);
    // for (int i = 0; i < image.rows; ++i) {
    //     for (int j = 0; j < image.cols; ++j) {
    //         if (image_harris.at<float>(i, j) < 0)
    //             edges.at<uchar>(i, j) = 255;
    //         else
    //             edges.at<uchar>(i, j) = 0;
    //     }
    // }
    // cv::imwrite("grayscale.png", image_intensity);
    // cv::imwrite("edges.png", edges);
    // // end debug

    cv::Mat result;
    std::unique_ptr<stereo::matcher::Matcher> matcher;
    try {
        if (num_threads > 1) {
            matcher = std::make_unique<stereo::matcher::MatcherMultiThreads>(
                image_yuv, image1_yuv, image_harris, image1_harris, max_disp,
                match_reward, start_gap_cost, continue_gap_cost, corner_ths,
                num_threads);
        } else {
            matcher = std::make_unique<stereo::matcher::MatcherSingleThread>(
                image_yuv, image1_yuv, image_harris, image1_harris, max_disp,
                match_reward, start_gap_cost, continue_gap_cost, corner_ths);
        }
        result = matcher->match();
        result *= output_multiplier;
    } catch (const std::invalid_argument &e) {
        std::cerr << e.what() << std::endl;
        exit(4);
    }

    std::string file_ext = ".png";
    std::string result_path = std::string(argv[3]);

    if (!std::equal(file_ext.rbegin(), file_ext.rend(), result_path.rbegin())) {
        result_path += file_ext;
    }

    if (!cv::imwrite(result_path, result)) {
        std::cerr << "Cannot save output in " << argv[3] << std::endl;
        exit(5);
    }

    return 0;
}
