#include <vector>
#include <ATen/ATen.h>
#include <c10/util/irange.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/Functions.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/cpu/moments_utils.h>

using at::Half;
using at::Tensor;
using phalf = at::Half;

template <typename T>
void sparse_im2col_cpu_kernel(
    const int n, const T *data_im, const int64_t *mask_h, const int64_t *mask_w, const int height,
    const int width, const int kernel_height, const int kernel_width, const int pad_height,
    const int pad_width, const int stride_height, const int stride_width,
    const int height_col,
    const int width_col, T *data_col, const int number) {
  at::parallel_for(0, n, 1, [&](int64_t start, int64_t end) {
    for (const auto index : c10::irange(start, end)) {  
    int64_t mask_index = index % number;

    int64_t w_out = mask_w[mask_index]; 

    int64_t h_out = mask_h[mask_index];
    int64_t channel_in = index / number; 
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
              ? im[i * width + j]             
              : static_cast<T>(0);
          col += number;      
        }
      }
  }
  });
}

void sparse_im2col_cuda(Tensor data_im, Tensor mask_x, Tensor mask_y,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            Tensor data_col,  const int number) {
  int height_col =
      (height + 2 * pad_h - ((ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - ((ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * number;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "sparse_im2col_cpu", ([&] {
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const int64_t *mask_h_ = mask_x.data_ptr<int64_t>();
        const int64_t *mask_w_ = mask_y.data_ptr<int64_t>();
        scalar_t *data_col_ = data_col.data_ptr<scalar_t>();

        sparse_im2col_cpu_kernel<scalar_t>(
            num_kernels, data_im_, mask_h_, mask_w_, height, width, ksize_h,
            ksize_w, pad_h, pad_w, stride_h, stride_w,
            height_col, width_col, data_col_, number);
      }));
}

template <typename scalar_t>
void sparse_tmp2output_cpu_kernel(const int n, const scalar_t *data_tmp_,
                                    const int height, const int width,
                                    const int channels,
                                    const int64_t *mask_h_,
                                    const int64_t *mask_w_,
                                    const int number, scalar_t *data_col_) {
  at::parallel_for(0, n, 1, [&](int64_t start, int64_t end) {
    for (const auto index : c10::irange(start, end)) {  
      const int m_index = index % number;
      const int h_im = mask_h_[m_index];
      const int w_im = mask_w_[m_index];
      const int c_im = index / number;
      data_col_[(c_im * height + h_im) * width + w_im] = data_tmp_[index];
  }
  });
}


void sparse_tmp2output(Tensor tmp, Tensor mask_x, Tensor mask_y, const int channels, const int height, const int width, const int number, Tensor col) {
  int num_kernels = channels * number;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      tmp.scalar_type(), "sparse_tmp2output_cpu", ([&] {
        const scalar_t *tmp_ = tmp.data_ptr<scalar_t>();
        const int64_t *mask_h_ = mask_x.data_ptr<int64_t>();
        const int64_t *mask_w_ = mask_y.data_ptr<int64_t>();
        scalar_t *col_ = col.data_ptr<scalar_t>();

        sparse_tmp2output_cpu_kernel<scalar_t>(
                num_kernels, tmp_, height, width, channels, mask_h_,
                mask_w_, number, col_);
      }));
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
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  const int64_t G = group;
  const int64_t D = C / G;
  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.data_ptr<T>() : nullptr;
  T* Y_data = Y.data_ptr<T>();
  T* mean_data = mean.data_ptr<T>();
  T* rstd_data = rstd.data_ptr<T>();
  const bool gamma_null = (gamma_data == nullptr);
  const bool beta_null = beta_data == nullptr;
  const int64_t inner_size = D * HxW;

  at::parallel_for(0, N * G, 1, [&](int64_t start, int64_t end) {
    for (const auto i : c10::irange(start, end)) {
      const T* X_ptr = X_data + i * inner_size;
      T mean_val = mean_data[i];
      T rstd_val = rstd_data[i];
      
      if (gamma_null && beta_null) {
        T* Y_ptr = Y_data + i * inner_size;
        for (const auto j : c10::irange(inner_size)) {
          Y_ptr[j] = (X_ptr[j] - mean_val) * rstd_val;
        }
      } else {
        const int64_t g = i % G;
        for (const auto j : c10::irange(D)) {
          const int64_t c = g * D + j;
          const T scale = rstd_val * (gamma_null ? T(1) : gamma_data[c]);
          const T bias = -scale * mean_val + (beta_null ? T(0) : beta_data[c]);
          X_ptr = X_data + (i * D + j) * HxW;
          T* Y_ptr = Y_data + (i * D + j) * HxW;
          for (const auto k : c10::irange(HxW)) {
            Y_ptr[k] = scale * X_ptr[k] + bias;
          }
        }
      }
      mean_data[i] = mean_val;
      rstd_data[i] = rstd_val;
    }
  });
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