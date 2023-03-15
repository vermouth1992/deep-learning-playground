#include <iostream>
#include <torch/torch.h>

#include <cuda.h>
#include <cuda_runtime.h>


__global__ void SigmoidForwardKernel(const int N, const float* X, float* Y){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < N) Y[i] = 1. / (1. + expf(-X[i]));
}


__global__ void SigmoidBackwardKernel(const int N, const float* dY, const float* Y, float* dX){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < N) dX[i] = dY[i] * Y[i] * (1. - Y[i]);
}



torch::Tensor d_sigmoid(torch::Tensor y, torch::Tensor dout) {
  torch::Tensor dx = (1. - y) * y * dout;
//   std::cout << y.size(0) << std::endl;
//     int64_t threads = 1024;
//     auto blocks = (y.size(0) + threads - 1) / threads;
//   SigmoidBackwardKernel<<<blocks, threads>>> (y.size(0), dout.data_ptr<float>(), y.data_ptr<float>(), dx.data_ptr<float>());
//
//   std::cout << dout << std::endl;
//   std::cout << y << std::endl;
//   std::cout << dx << std::endl;
  return dx;
}

torch::Tensor sigmoid_forward(torch::Tensor x) {
    torch::Tensor out = torch::empty_like(x);
    int64_t threads = 1024;
    auto blocks = (x.size(0) + threads - 1) / threads;
    SigmoidForwardKernel<<<blocks, threads>>> (x.size(0), x.data_ptr<float>(), out.data_ptr<float>());
    return out;
}



TORCH_LIBRARY(my_ops, m) {
  m.def("sigmoid_forward(Tensor x) -> Tensor");
  m.def("sigmoid_backward(Tensor y, Tensor dout) -> Tensor");
}


TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
  m.impl("sigmoid_forward", sigmoid_forward);
  m.impl("sigmoid_backward", d_sigmoid);
}