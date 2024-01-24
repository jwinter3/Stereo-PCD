#ifndef STEREO_MATCH
#define STEREO_MATCH

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>

namespace stereo {
namespace matcher {

class Matcher {
protected:
    const cv::Mat &left_;
    const cv::Mat &right_;

    const cv::Mat &left_harris_;
    const cv::Mat &right_harris_;

    cv::Mat disparity_;

    float match_reward_;
    float start_gap_cost_;
    float continue_gap_cost_;
    float edge_ths_;

    int max_disp_;

    double min_harris_;

    float color_score(int line, int left_column, int right_column);
    void reset_f_matrix(cv::Mat &f_matrix);
    void find_trace(int line, cv::Mat &f_matrix);
    void match_line(int line, cv::Mat &f_matrix,
                    std::vector<int> &y_gap_lenght);
    void fill_gaps(int line);
    float gap_cost(int gap_length);
    bool is_edge_left(int line, int column);
    bool is_edge_right(int line, int column);
    float pixel_error(const uchar &px1, const uchar &px2);
    template <class T, int n>
    float pixel_error(const cv::Vec<T, n> &px1, const cv::Vec<T, n> &px2);

public:
    Matcher(const cv::Mat &left, const cv::Mat &right,
            const cv::Mat &left_harris, const cv::Mat &right_harris,
            int max_disp, float match_reward, float start_gap_cost,
            float continue_gap_cost, float edge_ths);
    Matcher(const cv::Mat &left, const cv::Mat &right,
            const cv::Mat &left_harris, const cv::Mat &right_harris,
            float min_dist, float focal, float baseline, float match_reward,
            float start_gap_cost, float continue_gap_cost, float edge_ths);
    virtual cv::Mat match() = 0;
    virtual ~Matcher() = default;
};
} // namespace matcher
} // namespace stereo

#endif // STEREO_MATCH
