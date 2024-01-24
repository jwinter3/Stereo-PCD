#ifndef DATA_TYPES
#define DATA_TYPES

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace boost;
using namespace boost::python;

std::pair<int, int> dtypeToCVType(const numpy::dtype &dtype);
std::pair<numpy::dtype, int> cv_type_to_dtype(int ocvType);

cv::Mat ndarray_to_mat(const numpy::ndarray &array);
numpy::ndarray mat_to_ndarray(const cv::Mat &image);

#endif // DATA_TYPES
