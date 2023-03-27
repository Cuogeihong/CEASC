#include <torch/extension.h>

#include <vector>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstring>
#include <ctime>

using namespace at;
using namespace std;
using namespace torch::indexing;

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) \
  TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_INPUT(x) \
  CHECK_CUDA(x);            \
  CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x) \
  CHECK_CPU(x);            \
  CHECK_CONTIGUOUS(x)
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void sparse_im2col_cuda(Tensor data_im, Tensor mask_x, Tensor mask_y,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            Tensor data_col, const int number);

void sparse_tmp2output(Tensor tmp, Tensor mask_x, Tensor mask_y, const int nInputPlane, const int inputHeight, const int inputWidth, const int number, Tensor col);

void sparse_gn_cuda(const Tensor& X, const Tensor& gamma, const Tensor& beta, 
                        int64_t N, int64_t C, int64_t HxW, int64_t group, double eps,
                        Tensor& Y, Tensor& mean, Tensor& rstd);


torch::Tensor sparse_gn(Tensor input, Tensor pw_mean, Tensor pw_rstd, Tensor weight, Tensor bias, double eps, int num_groups) {
  
  const int N = input.size(0);
  const int C = input.size(1);
  const int HxW = input.size(2);

  auto memory_format = input.device().is_cpu() ?
      input.suggest_memory_format() : at::MemoryFormat::Contiguous;

  Tensor output = at::native::empty_like(
      input,
      c10::nullopt /* dtype */,
      c10::nullopt /* layout */,
      c10::nullopt /* device */,
      c10::nullopt /* pin_memory */,
      memory_format);  
  sparse_gn_cuda(input, weight, bias, N, C, HxW, num_groups, eps, output, pw_mean, pw_rstd);
  
  return output;
}


std::vector<Tensor> my_sparse_cpu_forward (
    Tensor input,
    Tensor hard,
    Tensor weights,
    Tensor bias,
    int stride,
    int padding,
    bool isbias,
    float base, 
    int num_groups,
    Tensor gnweight,
    Tensor gnbias,
    Tensor pw_mean,
    Tensor pw_rstd,
    float eps,
    Tensor nonzero_hard_x,
    Tensor nonzero_hard_y) {
  
  int kW = weights.size(2);
  int kH = weights.size(3);
  int dH = stride;
  int dW = stride;
  int padH = padding;
  int padW = padding;
  
  CHECK_CPU_INPUT(input);
  CHECK_CPU_INPUT(hard);
  CHECK_CPU_INPUT(weights);
  CHECK_CPU_INPUT(bias);

  at::DeviceGuard guard(input.device());

  int batch = 1;
  if (input.ndimension() == 3) {
    batch = 0;
    input.unsqueeze_(0);
    hard.unsqueeze_(0);
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  const long inputHeight = input.size(2);
  const long inputWidth = input.size(3);

  long nOutputPlane = weights.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - ((kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - ((kH - 1) + 1)) / dH + 1;

  TORCH_CHECK((hard.size(0) == batchSize), "invalid batch size of hard");
  auto output = torch::ones({batchSize, nOutputPlane, outputHeight, outputWidth}, input.options());
  output = output * base;

  for (int elt = 0; elt < batchSize; elt++) {

    auto mask_x = nonzero_hard_x;
    auto mask_y = nonzero_hard_y; 

    int number = mask_x.numel();
    if (number == 0) {
      continue;
    }
        
    auto columns = at::empty({nInputPlane * kW * kH, number}, input.options());

    sparse_im2col_cuda(input[elt], mask_x, mask_y, nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW, columns, number);
    columns = columns.view({columns.size(0), columns.size(1)});
    if (isbias) {
      auto tmp = torch::addmm(bias.index({Slice(), None}), weights.flatten(1), columns, 1, 1);
      if (num_groups != -999) {
        tmp = tmp.unsqueeze(0);
        tmp = sparse_gn(tmp, pw_mean, pw_rstd, gnweight, gnbias, eps, num_groups);
        tmp = tmp.squeeze(0);
      }
      sparse_tmp2output(tmp, mask_x, mask_y, nOutputPlane, outputHeight, outputWidth, number, output[elt]);
    } 
    else {
      auto tmp = torch::mm(weights.flatten(1), columns);
      if (num_groups != -999) {
        tmp = tmp.unsqueeze(0);
        tmp = sparse_gn(tmp, pw_mean, pw_rstd, gnweight, gnbias, eps, num_groups);
        tmp = tmp.squeeze(0);
      }
      sparse_tmp2output(tmp, mask_x, mask_y, nOutputPlane, outputHeight, outputWidth, number, output[elt]);
    }
  }

  output = output.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  if (batch == 0) {
    output = output.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    hard = hard.view({hard.size(1), hard.size(2), hard.size(3)});
  }
  return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &my_sparse_cpu_forward, "SPARSE_CONV forward (CPU)");
}