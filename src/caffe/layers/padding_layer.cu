#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

template <typename Dtype>
inline __device__ Dtype caffe_gpu_atomic_add(const Dtype val, Dtype* address);

template <>
inline __device__
float caffe_gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

// double atomicAdd implementation taken from:
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#axzz3PVCpVsEG
template <>
inline __device__
double caffe_gpu_atomic_add(const double val, double* address) {
  unsigned long long int* address_as_ull =  // NOLINT(runtime/int)
      // NOLINT_NEXT_LINE(runtime/int)
      reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;  // NOLINT(runtime/int)
  unsigned long long int assumed;  // NOLINT(runtime/int)
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

}  // namespace caffe


namespace caffe {


/*
Efficiency note:
The forward and backward processes are just default implementations
taken from the cpu version. The correct way of doing them would probably
be to run a specific kernel for each part of the padding grid 0-8.
To do this effectively without writing 9 kernels for each one you can use
a templated function of the boxId and call the kernels in a loop
from the cpu.
See here for reference of templated function for switch statements:
http://stackoverflow.com/questions/6179295/if-statement-inside-a-cuda-kernel
*/

template <typename Dtype>
__global__ void RepeatPadForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int padded_height, const int padded_width, 
    const int pad_l, const int pad_r, const int pad_t, const int pad_b,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(padded_index, nthreads) {
    const int pw = padded_index % padded_width;
    const int ph = (padded_index / padded_width) % padded_height;
    const int c = (padded_index / padded_width / padded_height) % channels;
    const int n = padded_index / padded_width / padded_height / channels;
    
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;       

    int boxIdx = ((ph<pad_t)?0:(ph<pad_t+height)?1:2)*3 + ((pw<pad_l)?0:(pw<pad_l+width)?1:2);
    int bottom_index = 0; 
    switch (boxIdx){
      case 0: bottom_index =          (0) * width +          (0); break; //Top Left 
      case 1: bottom_index =          (0) * width + (pw - pad_l); break; //Top Center
      case 2: bottom_index =          (0) * width +  (width - 1); break; //Top Right
      case 3: bottom_index = (ph - pad_t) * width +          (0); break; //Center Left
      case 4: bottom_index = (ph - pad_t) * width + (pw - pad_l); break; //Center Center
      case 5: bottom_index = (ph - pad_t) * width +  (width - 1); break; //Center Right
      case 6: bottom_index = (height - 1) * width +          (0); break; //Bottom Left
      case 7: bottom_index = (height - 1) * width + (pw - pad_l); break; //Bottom Center
      case 8: bottom_index = (height - 1) * width +  (width - 1); break; //Bottom Right
    }
            
    top_data[padded_index] = bottom_slice[bottom_index];
  }
}

template <typename Dtype>
__global__ void ConstantPadForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int padded_height, const int padded_width, 
    const int pad_l, const int pad_r, const int pad_t, const int pad_b, const float pad_value,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(padded_index, nthreads) {
    const int pw = padded_index % padded_width;
    const int ph = (padded_index / padded_width) % padded_height;
    const int c = (padded_index / padded_width / padded_height) % channels;
    const int n = padded_index / padded_width / padded_height / channels;
    
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;       

    int boxIdx = ((ph<pad_t)?0:(ph<pad_t+height)?1:2)*3 + ((pw<pad_l)?0:(pw<pad_l+width)?1:2);
    int bottom_index = 0; 
    switch (boxIdx){
      //case 0: bottom_index =          (0) * width +          (0); break; //Top Left 
      //case 1: bottom_index =          (0) * width + (pw - pad_l); break; //Top Center
      //case 2: bottom_index =          (0) * width +  (width - 1); break; //Top Right
      //case 3: bottom_index = (ph - pad_t) * width +          (0); break; //Center Left
      case 4: bottom_index = (ph - pad_t) * width + (pw - pad_l); 
         top_data[padded_index] = bottom_slice[bottom_index];        break; //Center Center
      //case 5: bottom_index = (ph - pad_t) * width +  (width - 1); break; //Center Right
      //case 6: bottom_index = (height - 1) * width +          (0); break; //Bottom Left
      //case 7: bottom_index = (height - 1) * width + (pw - pad_l); break; //Bottom Center
      //case 8: bottom_index = (height - 1) * width +  (width - 1); break; //Bottom Right
      default: top_data[padded_index] = pad_value;               break; 
    }
  }
}


template <typename Dtype>
void PaddingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();

  switch (this->layer_param_.padding_param().pad_method()) {
  case PaddingParameter_PadMethod_REPEAT:
    // NOLINT_NEXT_LINE(whitespace/operators)
    RepeatPadForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, padded_height_, padded_width_, 
        pad_l_, pad_r_, pad_t_, pad_b_, top_data);
    break;
  case PaddingParameter_PadMethod_CONSTANT:
    // NOLINT_NEXT_LINE(whitespace/operators)
    ConstantPadForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, padded_height_, padded_width_, 
        pad_l_, pad_r_, pad_t_, pad_b_, pad_value_, top_data);
    break;
  default:
    LOG(FATAL) << "Unknown padding method (GPU).";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void RepeatPadBackward(const int nthreads,
    const Dtype* const top_diff, const int num, const int channels,
    const int height, const int width, const int padded_height, const int padded_width, 
    const int pad_l, const int pad_r, const int pad_t, const int pad_b,
    Dtype* const bottom_diff) {

  CUDA_KERNEL_LOOP(padded_index, nthreads) {
    const int pw = padded_index % padded_width;
    const int ph = (padded_index / padded_width) % padded_height;
    const int c = (padded_index / padded_width / padded_height) % channels;
    const int n = padded_index / padded_width / padded_height / channels;
    
    Dtype* const bottom_diff_slice =
        bottom_diff + (n * channels + c) * height * width;       

    int bottom_index = 0;

    int boxIdx = ((ph<pad_t)?0:(ph<pad_t+height)?1:2)*3 + ((pw<pad_l)?0:(pw<pad_l+width)?1:2);
    switch (boxIdx){
      case 0: bottom_index =          (0) * width +          (0); break; //Top Left 
      case 1: bottom_index =          (0) * width + (pw - pad_l); break; //Top Center
      case 2: bottom_index =          (0) * width +  (width - 1); break; //Top Right
      case 3: bottom_index = (ph - pad_t) * width +          (0); break; //Center Left
      case 4: bottom_index = (ph - pad_t) * width + (pw - pad_l); break; //Center Center
      case 5: bottom_index = (ph - pad_t) * width +  (width - 1); break; //Center Right
      case 6: bottom_index = (height - 1) * width +          (0); break; //Bottom Left
      case 7: bottom_index = (height - 1) * width + (pw - pad_l); break; //Bottom Center
      case 8: bottom_index = (height - 1) * width +  (width - 1); break; //Bottom Right
    }
    //bottom_diff_slice[bottom_index] += top_diff[padded_index];  
    caffe_gpu_atomic_add(top_diff[padded_index],bottom_diff_slice + bottom_index);
  }
}

template <typename Dtype>
__global__ void ConstantPadBackward(const int nthreads,
    const Dtype* const top_diff, const int num, const int channels,
    const int height, const int width, const int padded_height, const int padded_width, 
    const int pad_l, const int pad_r, const int pad_t, const int pad_b,
    Dtype* const bottom_diff) {

  CUDA_KERNEL_LOOP(padded_index, nthreads) {
    const int pw = padded_index % padded_width;
    const int ph = (padded_index / padded_width) % padded_height;
    const int c = (padded_index / padded_width / padded_height) % channels;
    const int n = padded_index / padded_width / padded_height / channels;
    
    Dtype* const bottom_diff_slice =
        bottom_diff + (n * channels + c) * height * width;       

    int bottom_index = 0;

    int boxIdx = ((ph<pad_t)?0:(ph<pad_t+height)?1:2)*3 + ((pw<pad_l)?0:(pw<pad_l+width)?1:2);
    switch (boxIdx){
    //case 0: bottom_index =           (0) * width_ +           (0); break; //Top Left 
    //case 1: bottom_index =           (0) * width_ + (pw - pad_l_); break; //Top Center
    //case 2: bottom_index =           (0) * width_ +  (width_ - 1); break; //Top Right
    //case 3: bottom_index = (ph - pad_t_) * width_ +           (0); break; //Center Left
      case 4: bottom_index = (ph - pad_t) * width + (pw - pad_l);           //Center Center
              bottom_diff_slice[bottom_index] += top_diff[padded_index];        break; 
    //case 5: bottom_index = (ph - pad_t_) * width_ +  (width_ - 1); break; //Center Right
    //case 6: bottom_index = (height_ - 1) * width_ +           (0); break; //Bottom Left
    //case 7: bottom_index = (height_ - 1) * width_ + (pw - pad_l_); break; //Bottom Center
    //case 8: bottom_index = (height_ - 1) * width_ +  (width_ - 1); break; //Bottom Right
      default: break; // I.e. derivative is zero and has no influence
    }  
  }
}


template <typename Dtype>
void PaddingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int bottom_count = bottom[0]->count();
  const int top_count = top[0]->count();
  caffe_gpu_set(bottom_count, Dtype(0.), bottom_diff);
 
  switch (this->layer_param_.padding_param().pad_method()) {
  case PaddingParameter_PadMethod_REPEAT:
    // NOLINT_NEXT_LINE(whitespace/operators)
    RepeatPadBackward<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(
        top_count, top_diff, top[0]->num(), channels_,
        height_, width_, padded_height_, padded_width_,
        pad_l_, pad_r_, pad_t_, pad_b_,
        bottom_diff);
    break;
  case PaddingParameter_PadMethod_CONSTANT:
    // NOLINT_NEXT_LINE(whitespace/operators)
    ConstantPadBackward<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(
        top_count, top_diff, top[0]->num(), channels_,
        height_, width_, padded_height_, padded_width_, 
        pad_l_, pad_r_, pad_t_, pad_b_, 
        bottom_diff);
    break;
  case PaddingParameter_PadMethod_WRAP:
    // NOLINT_NEXT_LINE(whitespace/operators)
    LOG(FATAL) << "WRAP not implemented";
    break;
  default:
    LOG(FATAL) << "Unknown padding method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PaddingLayer);


}  // namespace caffe
