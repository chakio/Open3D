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
#include "pybind/pybind_utils.h"

namespace open3d {
namespace ml {

// Converts a py::array to a contiguous Tensor.
//
// If the input is contiguous, there won't be a copy, otherwise, the tensor will
// be copied to convert to contiguous. Tensor allows easy manipulations such as
// checking dtype and converting to contiguous array.
core::Tensor PyArrayToContiguousTensor(py::array array) {
    py::buffer_info info = array.request();
    core::SizeVector shape(info.shape.begin(), info.shape.end());
    core::SizeVector strides(info.strides.begin(), info.strides.end());
    for (size_t i = 0; i < strides.size(); ++i) {
        strides[i] /= info.itemsize;
    }
    core::Dtype dtype = pybind_utils::ArrayFormatToDtype(info.format);
    core::Device device("CPU:0");
    std::function<void(void*)> deleter = [](void*) -> void {};
    auto blob = std::make_shared<core::Blob>(device, info.ptr, deleter);
    return core::Tensor(shape, strides, info.ptr, dtype, blob).Contiguous();
}

py::array TensorToPyArray(const core::Tensor& tensor) {
    if (tensor.GetDevice().GetType() != core::Device::DeviceType::CPU) {
        utility::LogError(
                "Can only convert CPU Tensor to numpy. Copy "
                "Tensor to CPU before converting to numpy.");
    }
    py::dtype py_dtype =
            py::dtype(pybind_utils::DtypeToArrayFormat(tensor.GetDtype()));
    py::array::ShapeContainer py_shape(tensor.GetShape());
    core::SizeVector strides = tensor.GetStrides();
    int64_t element_byte_size = tensor.GetDtype().ByteSize();
    for (auto& s : strides) {
        s *= element_byte_size;
    }
    py::array::StridesContainer py_strides(strides);

    // `base_tensor` is a shallow copy of `tensor`. `base_tensor`
    // is on the heap and is owned by py::capsule
    // `base_tensor_capsule`. The capsule is referenced as the
    // "base" of the numpy tensor returned by o3d.Tensor.numpy().
    // When the "base" goes out-of-scope (e.g. when all numpy
    // tensors referencing the base have gone out-of-scope), the
    // deleter is called to free the `base_tensor`.
    //
    // This behavior is important when the origianl `tensor` goes
    // out-of-scope while we still want to keep the data alive.
    // e.g.
    //
    // ```python
    // def get_np_tensor():
    //     o3d_t = o3d.Tensor(...)
    //     return o3d_t.numpy()
    //
    // # Now, `o3d_t` is out-of-scope, but `np_t` still
    // # references the base tensor which references the
    // # underlying data of `o3d_t`. Thus np_t is still valid.
    // # When np_t goes out-of-scope, the underlying data will be
    // # finally freed.
    // np_t = get_np_tensor()
    // ```
    //
    // See:
    // https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
    core::Tensor* base_tensor = new core::Tensor(tensor);

    // See PyTorch's torch/csrc/Module.cpp
    auto capsule_destructor = [](PyObject* data) {
        core::Tensor* base_tensor = reinterpret_cast<core::Tensor*>(
                PyCapsule_GetPointer(data, "open3d::core::Tensor"));
        if (base_tensor) {
            delete base_tensor;
        } else {
            PyErr_Clear();
        }
    };

    py::capsule base_tensor_capsule(base_tensor, "open3d::core::Tensor",
                                    capsule_destructor);
    return py::array(py_dtype, py_shape, py_strides, tensor.GetDataPtr(),
                     base_tensor_capsule);
}

void pybind_contrib(py::module& m) {
    py::module m_contrib = m.def_submodule("contrib");

    m_contrib.def(
            "subsample",
            [](py::array points, py::array features, py::array classes,
               float sampleDl, int verbose) {
                std::vector<contrib::PointXYZ> original_points;
                std::vector<contrib::PointXYZ> subsampled_points;
                std::vector<float> original_features;
                std::vector<float> subsampled_features;
                std::vector<int> original_classes;
                std::vector<int> subsampled_classes;

                // Fill original_points.
                core::Tensor points_t = PyArrayToContiguousTensor(points);
                if (points_t.GetDtype() != core::Dtype::Float32) {
                    utility::LogError("points must be np.float32.");
                }
                if (points_t.NumDims() != 2 || points_t.GetShape()[1] != 3) {
                    utility::LogError(
                            "points must have shape (N, 3), but got {}.",
                            points_t.GetShape().ToString());
                }
                int64_t num_points = points_t.NumElements() / 3;
                original_points = std::vector<contrib::PointXYZ>(
                        reinterpret_cast<contrib::PointXYZ*>(
                                points_t.GetDataPtr()),
                        reinterpret_cast<contrib::PointXYZ*>(
                                points_t.GetDataPtr()) +
                                num_points);

                // Fill original_features.
                if (features != py::none()) {
                    core::Tensor features_t =
                            PyArrayToContiguousTensor(features);
                    if (features_t.GetDtype() != core::Dtype::Float32) {
                        utility::LogError("features must be np.float32.");
                    }
                    if (features_t.NumDims() != 2) {
                        utility::LogError(
                                "features must have shape (N, d), but got {}.",
                                features_t.GetShape().ToString());
                    }
                    if (features_t.GetShape()[0] != num_points) {
                        utility::LogError(
                                "features's shape {} is not compatible with "
                                "points's shape {}, their first dimension must "
                                "be equal.",
                                features_t.GetShape().ToString(),
                                points_t.GetShape().ToString());
                    }
                    original_features = features_t.ToFlatVector<float>();
                }

                // Fill original_classes.
                if (classes != py::none()) {
                    core::Tensor classes_t = PyArrayToContiguousTensor(classes);
                    if (classes_t.GetDtype() != core::Dtype::Int32) {
                        utility::LogError("classes must be np.int32.");
                    }
                    if (classes_t.NumDims() != 1) {
                        utility::LogError(
                                "classes must have shape (N,), but got {}.",
                                classes_t.GetShape().ToString());
                    }
                    if (classes_t.GetShape()[0] != num_points) {
                        utility::LogError(
                                "classes's shape {} is not compatible with "
                                "points's shape {}, their first dimension must "
                                "be equal.",
                                classes_t.GetShape().ToString(),
                                points_t.GetShape().ToString());
                    }
                    original_classes = classes_t.ToFlatVector<int>();
                }

                // Call function.
                grid_subsampling(original_points, subsampled_points,
                                 original_features, subsampled_features,
                                 original_classes, subsampled_classes, sampleDl,
                                 verbose);

                // Wrap subsampled_points as numpy array.

                return py::make_tuple(subsampled_points, subsampled_features,
                                      subsampled_classes);
            },
            "points"_a, "features"_a = py::none(), "classes"_a = py::none(),
            "sampleDl"_a = 0.1f, "verbose"_a = 0);
}

}  // namespace ml
}  // namespace open3d
