#include "PythonStereoMatch.h"
#include "DataTypes.h"

Images::Images(cv::Mat left_mat, cv::Mat right_mat, cv::Mat left_harris_mat,
               cv::Mat right_harris_mat)
    : left_mat(left_mat), right_mat(right_mat),
      left_harris_mat(left_harris_mat), right_harris_mat(right_harris_mat) {}

MatcherSingleThreadPython::MatcherSingleThreadPython(
    const numpy::ndarray &left, const numpy::ndarray &right,
    const numpy::ndarray &left_harris, const numpy::ndarray &right_harris,
    int max_disp, float match_reward, float start_gap_cost,
    float continue_gap_cost, float edge_ths)
    : images_(ndarray_to_mat(left), ndarray_to_mat(right),
              ndarray_to_mat(left_harris), ndarray_to_mat(right_harris)),
      matcher_(images_.left_mat, images_.right_mat, images_.left_harris_mat,
               images_.right_harris_mat, max_disp, match_reward, start_gap_cost,
               continue_gap_cost, edge_ths) {}

MatcherSingleThreadPython::MatcherSingleThreadPython(
    const numpy::ndarray &left, const numpy::ndarray &right,
    const numpy::ndarray &left_harris, const numpy::ndarray &right_harris,
    float min_dist, float focal, float baseline, float match_reward,
    float start_gap_cost, float continue_gap_cost, float edge_ths)
    : images_(ndarray_to_mat(left), ndarray_to_mat(right),
              ndarray_to_mat(left_harris), ndarray_to_mat(right_harris)),
      matcher_(images_.left_mat, images_.right_mat, images_.left_harris_mat,
               images_.right_harris_mat, min_dist, focal, baseline,
               match_reward, start_gap_cost, continue_gap_cost, edge_ths) {}

numpy::ndarray MatcherSingleThreadPython::match() {
    return mat_to_ndarray(matcher_.match());
}

MatcherMultiThreadsPython::MatcherMultiThreadsPython(
    const numpy::ndarray &left, const numpy::ndarray &right,
    const numpy::ndarray &left_harris, const numpy::ndarray &right_harris,
    int max_disp, float match_reward, float start_gap_cost,
    float continue_gap_cost, float edge_ths, int num_threads)
    : images_(ndarray_to_mat(left), ndarray_to_mat(right),
              ndarray_to_mat(left_harris), ndarray_to_mat(right_harris)),
      matcher_(images_.left_mat, images_.right_mat, images_.left_harris_mat,
               images_.right_harris_mat, max_disp, match_reward, start_gap_cost,
               continue_gap_cost, edge_ths, num_threads) {}

MatcherMultiThreadsPython::MatcherMultiThreadsPython(
    const numpy::ndarray &left, const numpy::ndarray &right,
    const numpy::ndarray &left_harris, const numpy::ndarray &right_harris,
    float min_dist, float focal, float baseline, float match_reward,
    float start_gap_cost, float continue_gap_cost, float edge_ths,
    int num_threads)
    : images_(ndarray_to_mat(left), ndarray_to_mat(right),
              ndarray_to_mat(left_harris), ndarray_to_mat(right_harris)),
      matcher_(images_.left_mat, images_.right_mat, images_.left_harris_mat,
               images_.right_harris_mat, min_dist, focal, baseline,
               match_reward, start_gap_cost, continue_gap_cost, edge_ths,
               num_threads) {}

numpy::ndarray MatcherMultiThreadsPython::match() {
    return mat_to_ndarray(matcher_.match());
}