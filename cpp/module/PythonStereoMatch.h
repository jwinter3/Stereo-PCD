#ifndef PYTHON_STEREO_MATCH
#define PYTHON_STEREO_MATCH

#include "StereoMatch/StereoMatchMultiThreads.h"
#include "StereoMatch/StereoMatchSingleThread.h"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

using namespace boost;
using namespace boost::python;

struct Images {
    const cv::Mat left_mat;
    const cv::Mat right_mat;
    const cv::Mat left_harris_mat;
    const cv::Mat right_harris_mat;

    Images(cv::Mat left_mat, cv::Mat right_mat, cv::Mat left_harris_mat,
           cv::Mat right_harris_mat);
};

class MatcherSingleThreadPython {
    Images images_;
    stereo::matcher::MatcherSingleThread matcher_;

public:
    MatcherSingleThreadPython(const numpy::ndarray &left,
                              const numpy::ndarray &right,
                              const numpy::ndarray &left_harris,
                              const numpy::ndarray &right_harris, int max_disp,
                              float match_reward, float start_gap_cost,
                              float continue_gap_cost, float edge_ths);
    MatcherSingleThreadPython(const numpy::ndarray &left,
                              const numpy::ndarray &right,
                              const numpy::ndarray &left_harris,
                              const numpy::ndarray &right_harris,
                              float min_dist, float focal, float baseline,
                              float match_reward, float start_gap_cost,
                              float continue_gap_cost, float edge_ths);

    numpy::ndarray match();
};

class MatcherMultiThreadsPython {
    Images images_;
    stereo::matcher::MatcherMultiThreads matcher_;

public:
    MatcherMultiThreadsPython(const numpy::ndarray &left,
                              const numpy::ndarray &right,
                              const numpy::ndarray &left_harris,
                              const numpy::ndarray &right_harris, int max_disp,
                              float match_reward, float start_gap_cost,
                              float continue_gap_cost, float edge_ths,
                              int num_threads = 4);
    MatcherMultiThreadsPython(const numpy::ndarray &left,
                              const numpy::ndarray &right,
                              const numpy::ndarray &left_harris,
                              const numpy::ndarray &right_harris,
                              float min_dist, float focal, float baseline,
                              float match_reward, float start_gap_cost,
                              float continue_gap_cost, float edge_ths,
                              int num_threads = 4);

    numpy::ndarray match();
};

#endif // PYTHON_STEREO_MATCH
