#include "StereoMatchMultiThreads.h"

#include <opencv2/imgproc.hpp>

using namespace stereo::matcher;

MatcherMultiThreads::MatcherMultiThreads(
    const cv::Mat &left, const cv::Mat &right, const cv::Mat &left_harris,
    const cv::Mat &right_harris, int max_disp, float match_reward,
    float start_gap_cost, float continue_gap_cost, float edge_ths,
    int num_threads)
    : Matcher(left, right, left_harris, right_harris, max_disp, match_reward,
              start_gap_cost, continue_gap_cost, edge_ths),
      num_threads_(num_threads), line_counter_(0) {}

MatcherMultiThreads::MatcherMultiThreads(
    const cv::Mat &left, const cv::Mat &right, const cv::Mat &left_harris,
    const cv::Mat &right_harris, float min_dist, float focal, float baseline,
    float match_reward, float start_gap_cost, float continue_gap_cost,
    float edge_ths, int num_threads)
    : Matcher(left, right, left_harris, right_harris, min_dist, focal, baseline,
              match_reward, start_gap_cost, continue_gap_cost, edge_ths),
      num_threads_(num_threads), line_counter_(0) {}

void MatcherMultiThreads::thread_match() {
    cv::Mat f_matrix(left_.cols + 1, max_disp_, CV_32FC1);
    std::vector<int> y_gap_length(left_.cols);
    reset_f_matrix(f_matrix);

    while (int line = ++line_counter_) {
        line--;
        if (line >= left_.rows) {
            break;
        }
        std::fill(y_gap_length.begin(), y_gap_length.end(), 0);
        match_line(line, f_matrix, y_gap_length);
        fill_gaps(line);
    }
}

cv::Mat MatcherMultiThreads::match() {
    for (int i = 0; i < num_threads_; ++i) {
        threads_.push_back(
            std::thread(&MatcherMultiThreads::thread_match, this));
    }

    for (auto &t : threads_) {
        t.join();
    }

    return disparity_;
}
