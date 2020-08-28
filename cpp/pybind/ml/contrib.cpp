// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/ml/contrib/GridSubsampling.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace ml {

void pybind_contrib(py::module &m) {
    m.def("subsample", [](py::array_t<float> points,
                          py::array_t<float> features, py::array_t<int> classes,
                          float sampleDl, int verbose) {
        std::vector<contrib::PointXYZ> original_points;
        std::vector<contrib::PointXYZ> subsampled_points;
        std::vector<float> original_features;
        std::vector<float> subsampled_features;
        std::vector<int> original_classes;
        std::vector<int> subsampled_classes;

        grid_subsampling(original_points, subsampled_points, original_features,
                         subsampled_features, original_classes,
                         subsampled_classes, sampleDl, verbose);

        return py::make_tuple(subsampled_points, subsampled_features,
                              subsampled_classes);
    });
}

}  // namespace ml
}  // namespace open3d
