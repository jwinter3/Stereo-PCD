#if defined(_MSC_VER) && (_MSC_VER >= 1400)
// disable msvc warnings for Boost.Python (Boost 1.63)
#pragma warning(disable : 4100)
#pragma warning(disable : 4244)
#endif

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "DataTypes.h"
#include "PythonStereoMatch.h"

using namespace boost;
using namespace boost::python;

void init_module() { numpy::initialize(); }

numpy::ndarray copy(numpy::ndarray a) {
    cv::Mat b = ndarray_to_mat(a);
    return mat_to_ndarray(b);
}

BOOST_PYTHON_MODULE(stereo_match) {
    init_module();

    def("copy", &copy);

    class_<MatcherSingleThreadPython, noncopyable>(
        "Matcher", init<numpy::ndarray, numpy::ndarray, numpy::ndarray,
                        numpy::ndarray, int, float, float, float, float>())
        .def(
            init<numpy::ndarray, numpy::ndarray, numpy::ndarray, numpy::ndarray,
                 float, float, float, float, float, float, float>())
        .def("match", &MatcherSingleThreadPython::match);

    class_<MatcherMultiThreadsPython, noncopyable>(
        "MultiThreadsMatcher",
        init<numpy::ndarray, numpy::ndarray, numpy::ndarray, numpy::ndarray,
             int, float, float, float, float, int>())
        .def(
            init<numpy::ndarray, numpy::ndarray, numpy::ndarray, numpy::ndarray,
                 float, float, float, float, float, float, float, int>())
        .def("match", &MatcherMultiThreadsPython::match);
}
