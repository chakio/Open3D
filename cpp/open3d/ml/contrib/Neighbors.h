#include <cstdint>
#include <set>

#include "open3d/ml/contrib/Cloud.h"

namespace open3d {
namespace ml {
namespace contrib {

void ordered_neighbors(std::vector<PointXYZ>& queries,
                       std::vector<PointXYZ>& supports,
                       std::vector<int>& neighbors_indices,
                       float radius);

void batch_ordered_neighbors(std::vector<PointXYZ>& queries,
                             std::vector<PointXYZ>& supports,
                             std::vector<int>& q_batches,
                             std::vector<int>& s_batches,
                             std::vector<int>& neighbors_indices,
                             float radius);

void batch_nanoflann_neighbors(std::vector<PointXYZ>& queries,
                               std::vector<PointXYZ>& supports,
                               std::vector<int>& q_batches,
                               std::vector<int>& s_batches,
                               std::vector<int>& neighbors_indices,
                               float radius);

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
