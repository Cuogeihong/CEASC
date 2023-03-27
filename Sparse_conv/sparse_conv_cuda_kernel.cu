#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <type_traits>
#include <thrust/tuple.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/block_reduce.cuh>

#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>

#include <iostream>

using namespace at;
using at::Half;
using at::Tensor;
using phalf = at::Half;

#define __PHALF(x) (x)


#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                             \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
       i += blockDim.x * gridDim.x)                                 \
    for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); \
         j += blockDim.y * gridDim.y)

#define CUDA_2D_KERNEL_BLOCK_LOOP(i, n, j, m)          \
  for (size_t i = blockIdx.x; i < (n); i += gridDim.x) \
    for (size_t j = blockIdx.y; j < (m); j += gridDim.y)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
  int optimal_block_num = (N + num_threads - 1) / num_threads;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}


template <typename T>
__global__ void sparse_im2col_gpu_kernel(
    const int n, const T *data_im, const int64_t *mask_h, const int64_t *mask_w, const int height,
    const int width, const int kernel_height, const int kernel_width, const int pad_height,
    const int pad_width, const int stride_height, const int stride_width,
    const int height_col,
    const int width_col, T *data_col, const int number) {
  CUDA_1D_KERNEL_LOOP(index, n) {

    int64_t mask_index = index % number;

    int64_t w_out = mask_w[mask_index]; 

    int64_t h_out = mask_h[mask_index];
    int64_t channel_in = index / number; // 512
    int64_t channel_out = channel_in * kernel_height * kernel_width;
    int64_t h_in = h_out * stride_height - pad_height;
    int64_t w_in = w_out * stride_width - pad_width;

    
    const T* im = data_im + (channel_in * height + h_in) * width + w_in;
    T* col = data_col + channel_out * number + mask_index;
    for (int64_t i = 0; i < kernel_height; ++i) {
      for (int64_t j = 0; j < kernel_width; ++j) {          
        int64_t h = h_in + i;
        int64_t w = w_in + j;
        *col = (h >= 0 && w >= 0 && h < height && w < width)
            ? im[i * width + j]             // data_im[channel_in][h_in + i][w_in + j]
            : static_cast<T>(0);
        col += number;      // data_col[channel_out + 1][h_out + i][w_out + j]
      }
    }
  }
}


void sparse_im2col_cuda(Tensor data_im, Tensor mask_x, Tensor mask_y,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            Tensor data_col,  const int number) {
  // num_axes should be smaller than block size
  // todo: check parallel_imgs is correctly passed in
  int height_col =
      (height + 2 * pad_h - ((ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - ((ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * number;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "sparse_im2col_gpu", ([&] {
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const int64_t *mask_h_ = mask_x.data_ptr<int64_t>();
        const int64_t *mask_w_ = mask_y.data_ptr<int64_t>();
        scalar_t *data_col_ = data_col.data_ptr<scalar_t>();


        sparse_im2col_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels),
                                       THREADS_PER_BLOCK, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_im_, mask_h_, mask_w_, height, width, ksize_h,
            ksize_w, pad_h, pad_w, stride_h, stride_w,
            height_col, width_col, data_col_, number);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}


template <typename scalar_t>
__global__ void sparse_tmp2output_gpu_kernel(const int n, const scalar_t *data_tmp_,
                                    const int height, const int width,
                                    const int channels,
                                    const int64_t *mask_h_,
                                    const int64_t *mask_w_,
                                    const int number, scalar_t *data_col_) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int m_index = index % number;
    const int h_im = mask_h_[m_index];
    const int w_im = mask_w_[m_index];
    const int c_im = index / number;
    data_col_[(c_im * height + h_im) * width + w_im] = data_tmp_[index];
  }
}


void sparse_tmp2output(Tensor tmp, Tensor mask_x, Tensor mask_y, const int channels, const int height, const int width, const int number, Tensor col) {
  int num_kernels = channels * number;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      tmp.scalar_type(), "sparse_tmp2output_gpu", ([&] {
        const scalar_t *tmp_ = tmp.data_ptr<scalar_t>();
        const int64_t *mask_h_ = mask_x.data_ptr<int64_t>();
        const int64_t *mask_w_ = mask_y.data_ptr<int64_t>();
        scalar_t *col_ = col.data_ptr<scalar_t>();

        sparse_tmp2output_gpu_kernel<scalar_t>
              <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
                num_kernels, tmp_, height, width, channels, mask_h_,
                mask_w_, number, col_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}



template <typename T>
__global__ void RowwiseMomentsCUDAKernel(
    int64_t N,
    T eps,
    const T* dw_x,
    T* mean,
    T* rstd) {
  using T_ACC = at::acc_type<T, true>;
  using WelfordType = at::native::WelfordData<T_ACC, int64_t, T_ACC>;
  using WelfordOp =
      at::native::WelfordOps<T_ACC, T_ACC, int64_t, T_ACC, thrust::pair<T_ACC, T_ACC>>;

  const int64_t i = blockIdx.x;
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    val = welford_op.reduce(val, static_cast<T_ACC>(dw_x[index]), index);
  }
  if (blockDim.x <= C10_WARP_SIZE) {
    val = at::native::cuda_utils::WarpReduce(val, welford_op);
  } else {
    __shared__ typename std::aligned_storage<
        sizeof(WelfordType),
        alignof(WelfordType)>::type val_shared[C10_WARP_SIZE];
    WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
    val = at::native::cuda_utils::BlockReduce(
        val,
        welford_op,
        /*identity_element=*/WelfordType(0, 0, 0, 0),
        val_shared_ptr);
  }
  if (threadIdx.x == 0) {
    T_ACC m1;
    T_ACC m2;
    thrust::tie(m2, m1) = welford_op.project(val);
    mean[i] = m1;
    rstd[i] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
  }
}

template <typename T>
__global__ void ComputeFusedParamsCUDAKernel(
    int64_t N,
    int64_t C,
    int64_t group,
    const T* mean,
    const T* rstd,
    const T* gamma,
    const T* beta,
    at::acc_type<T, true>* a,
    at::acc_type<T, true>* b) {
  using T_ACC = at::acc_type<T, true>;
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * C) {
    const int64_t ng = index / (C / group);
    const int64_t c = index % C;
    const T_ACC scale = static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(gamma[c]);
    a[index] = scale;
    b[index] = -scale * static_cast<T_ACC>(mean[ng]) + static_cast<T_ACC>(beta[c]);
  }
}


template <typename T>
void GroupNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    T eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  using T_ACC = at::acc_type<T, true>;
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  const int64_t G = group;
  const int64_t D = C / G;
  const T* X_data = X.data_ptr<T>();
  T* Y_data = Y.data_ptr<T>();
  T* rstd_data = rstd.data_ptr<T>();
  T* mean_data = mean.data_ptr<T>();
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  

  const auto kAccType =
      (X.scalar_type() == at::kHalf || X.scalar_type() == at::kBFloat16)
      ? at::kFloat
      : X.scalar_type();
  Tensor a = at::empty({N, C}, X.options().dtype(kAccType));
  Tensor b = at::empty({N, C}, X.options().dtype(kAccType));
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  T_ACC* a_data = a.data_ptr<T_ACC>();
  T_ACC* b_data = b.data_ptr<T_ACC>();
  // TODO: Since there is some issues in gpu_kernel_multiple_outputs, we are
  // using maunal kernel here. Make it using gpu_kernel_multiple_outputs once
  // the issue fixed.
  const int64_t kCUDANumThreads = 512;
  const int64_t B = (N * C + kCUDANumThreads - 1) / kCUDANumThreads;
  ComputeFusedParamsCUDAKernel<T><<<B, kCUDANumThreads, 0, cuda_stream>>>(
      N, C, G, mean_data, rstd_data, gamma_data, beta_data, a_data, b_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(std::is_same<T, T_ACC>::value)
                  .resize_outputs(false)
                  .add_owned_output(Y.view({N * C, HxW}))
                  .add_owned_input(X.view({N * C, HxW}))
                  .add_owned_input(a.view({N * C, 1}))
                  .add_owned_input(b.view({N * C, 1}))
                  .build();
  at::native::gpu_kernel(iter, [] GPU_LAMBDA(T x, T_ACC a, T_ACC b) -> T {
    return a * static_cast<T_ACC>(x) + b;
  });

  AT_CUDA_CHECK(cudaGetLastError());
}


void sparse_gn_cuda(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "GroupNormKernelImpl",
      [&]() {
        GroupNormKernelImplInternal<scalar_t>(
            X,
            gamma,
            beta,
            N,
            C,
            HxW,
            group,
            static_cast<scalar_t>(eps),
            Y,
            mean,
            rstd);
      });
}
