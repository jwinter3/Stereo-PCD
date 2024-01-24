#ifndef STEREO_MATCH_THREADS
#define STEREO_MATCH_THREADS

#include "StereoMatch.h"

#include <atomic>
#include <thread>

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>

namespace stereo {
namespace matcher {

class MatcherMultiThreads : public Matcher {
    int num_threads_;
    std::atomic_int line_counter_;

    std::vector<std::thread> threads_;

    void thread_match();

public:
    MatcherMultiThreads(const cv::Mat &left, const cv::Mat &right,
                        const cv::Mat &left_harris, const cv::Mat &right_harris,
                        int max_disp, float match_reward, float start_gap_cost,
                        float continue_gap_cost, float edge_ths,
                        int num_threads = 4);
    MatcherMultiThreads(const cv::Mat &left, const cv::Mat &right,
                        const cv::Mat &left_harris, const cv::Mat &right_harris,
                        float min_dist, float focal, float baseline,
                        float match_reward, float start_gap_cost,
                        float continue_gap_cost, float edge_ths,
                        int num_threads = 4);

    cv::Mat match();
};
} // namespace matcher
} // namespace stereo

#endif // STEREO_MATCH_THREADS
