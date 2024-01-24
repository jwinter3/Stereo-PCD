#include "StereoMatch.h"

#include <opencv2/imgproc.hpp>

using namespace stereo::matcher;

Matcher::Matcher(const cv::Mat &left, const cv::Mat &right,
                 const cv::Mat &left_harris, const cv::Mat &right_harris,
                 int max_disp, float match_reward, float start_gap_cost,
                 float continue_gap_cost, float edge_ths)
    : left_(left), right_(right), left_harris_(left_harris),
      right_harris_(right_harris), edge_ths_(edge_ths), max_disp_(max_disp) {
    if (left.rows != right.rows) {
        throw std::invalid_argument("Images must have the same dimensions");
    }
    if (left.cols != right.cols) {
        throw std::invalid_argument("Images must have the same dimensions");
    }
    if (left.channels() != right.channels()) {
        throw std::invalid_argument(
            "Images must have the same number of channels");
    }
    if (left.channels() != 1 && left.channels() != 3) {
        throw std::invalid_argument("Images must have 1 or 3 channels");
    }
    if (left.depth() != CV_8U || right.depth() != CV_8U) {
        throw std::invalid_argument("Images must have elements of type CV_8U");
    }

    if (max_disp <= 1 || max_disp >= left_.cols) {
        max_disp_ = std::max(2, left_.cols - 1);
    }

    disparity_ = cv::Mat::zeros(left.rows, left.cols, CV_16UC1);

    int gap_cost_norm = left.channels() *
                        static_cast<int>(std::numeric_limits<uchar>::max()) *
                        static_cast<int>(std::numeric_limits<uchar>::max());

    start_gap_cost_ = abs(start_gap_cost) * start_gap_cost * gap_cost_norm;

    continue_gap_cost_ =
        abs(continue_gap_cost) * continue_gap_cost * gap_cost_norm;

    match_reward_ = abs(match_reward) * match_reward * gap_cost_norm;

    double min_left, min_right;

    cv::minMaxIdx(left_harris_, &min_left);
    cv::minMaxIdx(right_harris_, &min_right);

    min_harris_ = std::max(min_left, min_right);
}

Matcher::Matcher(const cv::Mat &left, const cv::Mat &right,
                 const cv::Mat &left_harris, const cv::Mat &right_harris,
                 float min_dist, float focal, float baseline,
                 float match_reward, float start_gap_cost,
                 float continue_gap_cost, float edge_ths)
    : Matcher(left, right, left_harris, right_harris,
              std::ceil(focal * baseline / min_dist), match_reward,
              start_gap_cost, continue_gap_cost, edge_ths) {}

float Matcher::pixel_error(const uchar &px1, const uchar &px2) {
    return static_cast<float>(px2 - px1) * static_cast<float>(px2 - px1);
}

template <class T, int n>
float Matcher::pixel_error(const cv::Vec<T, n> &px1, const cv::Vec<T, n> &px2) {
    float error = 0;
    for (int i = 0; i < n; ++i) {
        error += (px2[i] - px1[i]) * (px2[i] - px1[i]);
    }

    return error;
}

float Matcher::color_score(int line, int left_column, int right_column) {
    if (left_.channels() == 3) {
        return pixel_error(left_.at<cv::Vec3b>(line, left_column),
                           right_.at<cv::Vec3b>(line, right_column));
    }

    return pixel_error(left_.at<uchar>(line, left_column),
                       right_.at<uchar>(line, right_column));
}

bool Matcher::is_edge_left(int line, int column) {
    return left_harris_.at<float>(line, column) < edge_ths_ * min_harris_ &&
           left_harris_.at<float>(line, column - 1) > edge_ths_ * min_harris_;
}

bool Matcher::is_edge_right(int line, int column) {
    return right_harris_.at<float>(line, column) < edge_ths_ * min_harris_ &&
           right_harris_.at<float>(line, column - 1) > edge_ths_ * min_harris_;
}

float Matcher::gap_cost(int gap_length) {
    if (gap_length == 0) {

        return start_gap_cost_;
    }
    if (gap_length == 1) {
        return continue_gap_cost_;
    }

    return 0;
}

void Matcher::reset_f_matrix(cv::Mat &f_matrix) {
    f_matrix.at<float>(0, 0) = 0;
    f_matrix.at<float>(0, 1) = start_gap_cost_;

    for (float *ptr = f_matrix.ptr<float>(0) + 2; ptr < f_matrix.ptr<float>(1);
         ++ptr) {
        *ptr = start_gap_cost_ + continue_gap_cost_;
    }
}

void Matcher::match_line(int line, cv::Mat &f_matrix,
                         std::vector<int> &y_gap_lenght) {
    for (int i = 0; i < f_matrix.rows - 2; ++i) {
        float score =
            f_matrix.at<float>(i, 0) - color_score(line, 0, i) + match_reward_;

        float y_gap_score =
            f_matrix.at<float>(i, 1) + gap_cost(y_gap_lenght[i]);

        if (y_gap_score >= score) {
            ++y_gap_lenght[i];
            f_matrix.at<float>(i + 1, 0) = y_gap_score;
        } else {
            y_gap_lenght[i] = 0;
            f_matrix.at<float>(i + 1, 0) = score;
        }

        int x_gap_length = 0;
        for (int j = 1; j < std::min(left_.cols - i, max_disp_) - 1; ++j) {
            float score = f_matrix.at<float>(i, j) -
                          color_score(line, i + j, i) + match_reward_;

            if (is_edge_left(line, i) && is_edge_right(line, i + j)) {
                score += match_reward_;
            }

            float y_gap_score =
                f_matrix.at<float>(i, j + 1) + gap_cost(y_gap_lenght[i + j]);
            float x_gap_score =
                f_matrix.at<float>(i + 1, j - 1) + gap_cost(x_gap_length);

            if (score > x_gap_score && score > y_gap_score) {
                f_matrix.at<float>(i + 1, j) = score;
                x_gap_length = 0;
                y_gap_lenght[i + j] = 0;
            } else if (x_gap_score >= y_gap_score) {
                f_matrix.at<float>(i + 1, j) = x_gap_score;
                ++x_gap_length;
                if (x_gap_score == y_gap_score) {
                    ++y_gap_lenght[i + j];
                } else {
                    y_gap_lenght[i + j] = 0;
                }
            } else {
                f_matrix.at<float>(i + 1, j) = y_gap_score;
                x_gap_length = 0;
                ++y_gap_lenght[i + j];
            }

            // Same output, but much more time
            // if (x_gap_score == max_score) {
            //     ++x_gap_length;
            // } else {
            //     x_gap_length = 0;
            // }

            // if (y_gap_score == max_score) {
            //     ++y_gap_length_[j];
            // } else {
            //     y_gap_length_[j] = 0;
            // }
        }
        int last = std::min(left_.cols - i, max_disp_) - 1;
        score = f_matrix.at<float>(i, last) - color_score(line, i, i + last) +
                match_reward_;
        float x_gap_score =
            f_matrix.at<float>(i + 1, last - 1) + gap_cost(x_gap_length);
        f_matrix.at<float>(i + 1, last) = std::max(score, x_gap_score);
    }

    f_matrix.at<float>(f_matrix.rows - 1, 0) =
        f_matrix.at<float>(f_matrix.rows - 2, 0);

    find_trace(line, f_matrix);
}

void Matcher::find_trace(int line, cv::Mat &f_matrix) {
    int i = f_matrix.rows - 1;
    int j = 0;

    while (i > 0) {
        float diag = f_matrix.at<float>(i - 1, j);
        float up = j < f_matrix.cols - 1
                       ? f_matrix.at<float>(i - 1, j + 1)
                       : -std::numeric_limits<float>::infinity();
        float left = j > 0 ? f_matrix.at<float>(i, j - 1)
                           : -std::numeric_limits<float>::infinity();

        if (diag >= up && diag >= left) {
            disparity_.at<unsigned short>(cv::Point(i + j, line)) = j;
            i--;
        } else if (up >= diag && up >= left) {
            i--;
            j++;
        } else {
            j--;
        }
    }
}

void Matcher::fill_gaps(int line) {
    int start = 0;
    int end = 0;
    unsigned short cur = 0;
    unsigned short prev = 0;

    for (int i = 0; i < disparity_.cols; ++i) {
        if (disparity_.at<unsigned short>(cv::Point(i, line)) != 0) {
            start = i;
            break;
        }
    }

    for (int i = disparity_.cols - 1; i >= 0; --i) {
        if (disparity_.at<unsigned short>(cv::Point(i, line)) != 0) {
            end = i;
            break;
        }
    }

    for (int i = end; i > start; --i) {

        cur = disparity_.at<unsigned short>(cv::Point(i, line));
        if (cur == 0) {
            disparity_.at<unsigned short>(cv::Point(i, line)) = prev;
        } else {
            prev = cur;
        }

        if (is_edge_left(line, i)) {
            prev = 0;
        }
    }

    for (int i = start; i < end; ++i) {

        cur = disparity_.at<unsigned short>(cv::Point(i, line));
        if (cur == 0) {
            disparity_.at<unsigned short>(cv::Point(i, line)) = prev;
        } else {
            prev = cur;
        }
    }
}
