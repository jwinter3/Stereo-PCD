#include "DataTypes.h"

#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>

std::pair<int, int> dtype_tocv_type(const numpy::dtype &dtype) {
    const std::vector<std::pair<numpy::dtype, std::pair<int, int>>> data_types =
        {
            {numpy::dtype::get_builtin<cv::uint8_t>(),
             {CV_8U, sizeof(cv::uint8_t)}},
            {numpy::dtype::get_builtin<cv::int8_t>(),
             {CV_8S, sizeof(cv::int8_t)}},
            {numpy::dtype::get_builtin<cv::uint16_t>(),
             {CV_16U, sizeof(cv::uint16_t)}},
            {numpy::dtype::get_builtin<cv::int16_t>(),
             {CV_16S, sizeof(cv::int16_t)}},
            {numpy::dtype::get_builtin<cv::int32_t>(),
             {CV_32S, sizeof(cv::int32_t)}},
            {numpy::dtype::get_builtin<float>(), {CV_32F, sizeof(float)}},
            {numpy::dtype::get_builtin<double>(), {CV_64F, sizeof(double)}},
            {numpy::dtype::get_builtin<cv::float16_t>(),
             {CV_16F, sizeof(cv::float16_t)}},
        };
    for (auto type : data_types) {
        if (numpy::equivalent(dtype, type.first)) {
            return type.second;
        }
    }

    throw std::runtime_error("Invalid data type");
}

std::pair<numpy::dtype, int> cv_type_to_dtype(int ocv_type) {
    const std::unordered_map<int, std::pair<numpy::dtype, int>> data_types = {
        {CV_8U,
         {numpy::dtype::get_builtin<cv::uint8_t>(), sizeof(cv::uint8_t)}},
        {CV_8S, {numpy::dtype::get_builtin<cv::int8_t>(), sizeof(cv::int8_t)}},
        {CV_16U,
         {numpy::dtype::get_builtin<cv::uint16_t>(), sizeof(cv::uint16_t)}},
        {CV_16S,
         {numpy::dtype::get_builtin<cv::int16_t>(), sizeof(cv::int16_t)}},
        {CV_32S,
         {numpy::dtype::get_builtin<cv::int32_t>(), sizeof(cv::int32_t)}},
        {CV_32F, {numpy::dtype::get_builtin<float>(), sizeof(float)}},
        {CV_64F, {numpy::dtype::get_builtin<double>(), sizeof(double)}},
        {CV_16F,
         {numpy::dtype::get_builtin<cv::float16_t>(), sizeof(cv::float16_t)}},
    };

    if (auto search = data_types.find(ocv_type); search != data_types.end()) {
        return search->second;
    }

    throw std::runtime_error("Invalid data type");
}

numpy::ndarray mat_to_ndarray(const cv::Mat &image) {
    cv::Size img_size = image.size();
    auto dtype_size = cv_type_to_dtype(image.depth());
    long int size =
        img_size.width * img_size.height * image.channels() * dtype_size.second;
    std::vector<Py_intptr_t> shape = {img_size.height, img_size.width,
                                      image.channels()};
    numpy::ndarray result =
        numpy::empty(image.dims + 1, shape.data(), dtype_size.first);
    memcpy(result.get_data(), image.ptr(), size);
    return result;
}

cv::Mat ndarray_to_mat(const numpy::ndarray &array) {
    int ndims = array.get_nd();
    if (ndims != 2 && ndims != 3) {
        throw std::runtime_error("Invalid shape od ndarray");
    }

    const Py_intptr_t *shape = array.get_shape();

    int rows = shape[0];
    int cols = shape[1];
    int channels = ndims == 3 ? shape[2] : 1;

    std::pair<int, int> depth_size = dtype_tocv_type(array.get_dtype());
    int depth = depth_size.first;
    int type = CV_MAKETYPE(depth, channels);

    cv::Mat image = cv::Mat(rows, cols, type, array.get_data());

    return image;
}