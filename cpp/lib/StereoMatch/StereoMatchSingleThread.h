#ifndef STEREO_MATCH_SINGLE
#define STEREO_MATCH_SINGLE

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>

#include "StereoMatch.h"

namespace stereo {
namespace matcher {

class MatcherSingleThread : public Matcher {
    cv::Mat f_matrix_;
    std::vector<int> y_gap_length_;

public:
    MatcherSingleThread(const cv::Mat &left, const cv::Mat &right,
                        const cv::Mat &left_harris, const cv::Mat &right_harris,
                        int max_disp, float match_reward, float start_gap_cost,
                        float continue_gap_cost, float edge_ths);
    MatcherSingleThread(const cv::Mat &left, const cv::Mat &right,
                        const cv::Mat &left_harris, const cv::Mat &right_harris,
                        float min_dist, float focal, float baseline,
                        float match_reward, float start_gap_cost,
                        float continue_gap_cost, float edge_ths);

    cv::Mat match();
};
} // namespace matcher
} // namespace stereo

#endif // STEREO_MATCH_SINGLE
