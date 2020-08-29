#include <cstdint>
#include <set>

#include "Cloud.h"

using namespace std;

void ordered_neighbors(vector<PointXYZ>& queries,
                       vector<PointXYZ>& supports,
                       vector<int>& neighbors_indices,
                       float radius);

void batch_ordered_neighbors(vector<PointXYZ>& queries,
                             vector<PointXYZ>& supports,
                             vector<int>& q_batches,
                             vector<int>& s_batches,
                             vector<int>& neighbors_indices,
                             float radius);

void batch_nanoflann_neighbors(vector<PointXYZ>& queries,
                               vector<PointXYZ>& supports,
                               vector<int>& q_batches,
                               vector<int>& s_batches,
                               vector<int>& neighbors_indices,
                               float radius);
