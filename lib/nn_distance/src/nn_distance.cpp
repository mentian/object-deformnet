#include <torch/extension.h>
#include <vector>

int nn_distance_cuda_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist1, at::Tensor dist2, at::Tensor idx1, at::Tensor idx2);


int nn_distance_cuda_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz1, at::Tensor gradxyz2, at::Tensor graddist1, at::Tensor graddist2, at::Tensor idx1, at::Tensor idx2);


int nn_distance_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist1, at::Tensor dist2, at::Tensor idx1, at::Tensor idx2) {
    return nn_distance_cuda_forward(xyz1, xyz2, dist1, dist2, idx1, idx2);
}


int nn_distance_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz1, at::Tensor gradxyz2, at::Tensor graddist1, 
					  	 at::Tensor graddist2, at::Tensor idx1, at::Tensor idx2) {
    return nn_distance_cuda_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &nn_distance_forward, "nn_distance forward (CUDA)");
  m.def("backward", &nn_distance_backward, "nn_distance backward (CUDA)");
}