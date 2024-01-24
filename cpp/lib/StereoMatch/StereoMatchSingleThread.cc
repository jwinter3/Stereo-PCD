#include "StereoMatchSingleThread.h"

#include <opencv2/imgproc.hpp>

using namespace stereo::matcher;

MatcherSingleThread::MatcherSingleThread(
    const cv::Mat &left, const cv::Mat &right, const cv::Mat &left_harris,
    const cv::Mat &right_harris, int max_disp, float match_reward,
    float start_gap_cost, float continue_gap_cost, float edge_ths)
    : Matcher(left, right, left_harris, right_harris, max_disp, match_reward,
              start_gap_cost, continue_gap_cost, edge_ths),
      y_gap_length_(left.cols) {
    f_matrix_ = cv::Mat(left_.cols + 1, max_disp_, CV_32FC1);

    reset_f_matrix(f_matrix_);
}

MatcherSingleThread::MatcherSingleThread(
    const cv::Mat &left, const cv::Mat &right, const cv::Mat &left_harris,
    const cv::Mat &right_harris, float min_dist, float focal, float baseline,
    float match_reward, float start_gap_cost, float continue_gap_cost,
    float edge_ths)
    : Matcher(left, right, left_harris, right_harris, min_dist, focal, baseline,
              match_reward, start_gap_cost, continue_gap_cost, edge_ths),
      y_gap_length_(left.cols) {
    f_matrix_ = cv::Mat(left_.cols + 1, max_disp_, CV_32FC1);

    reset_f_matrix(f_matrix_);
}

cv::Mat MatcherSingleThread::match() {
    for (int i = 0; i < left_.rows; ++i) {
        std::fill(y_gap_length_.begin(), y_gap_length_.end(), 0);
        match_line(i, f_matrix_, y_gap_length_);
        fill_gaps(i);
    }

    return disparity_;
}
