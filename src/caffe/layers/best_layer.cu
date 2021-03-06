#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"



namespace caffe {
    
    // CUDA kernele for forward
    template <typename Dtype>
    __global__ void BestForward(const int n, const int channels, const int dim,
                                const Dtype* in, Dtype* out, const Dtype* beta,
                                const int div_factor) {
        CUDA_KERNEL_LOOP(index, n) {
            int c = (index / dim) % channels / div_factor;
            out[index] = max(in[index] - beta[c], 0.0f) - max(-in[index] - beta[c], 0.0f);
        }
    }
    
    // CUDA kernel for bottom backward
    template <typename Dtype>
    __global__ void BestBackward(const int n, const int channels, const int dim,
                                 const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff,
                                 const Dtype* beta, const int div_factor) {
        CUDA_KERNEL_LOOP(index, n) {
            int c = (index / dim) % channels / div_factor;
            out_diff[index] = in_diff[index] * (    (abs(in_data[index]) > abs(beta[c]))
                                                + 2.0f*(abs(in_data[index]) < -beta[c]) );
        }
    }
    
    // CUDA kernel for element-wise parameter backward
    template <typename Dtype>
    __global__ void BestParamBackward(const int n, const int channels, const int dim,
                                      const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff,
                                      const Dtype* beta, const int div_factor) {
        CUDA_KERNEL_LOOP(index, n) {
            int c = (index / dim) % channels / div_factor;
            out_diff[index] = in_diff[index] * (- (fabsf(beta[c]) <  in_data[index])
                                                + (fabsf(beta[c]) < -in_data[index]));
        }
    }
    
    template <typename Dtype>
    void BestLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        const int count = bottom[0]->count();
        const int dim = bottom[0]->count(2);
        const int channels = bottom[0]->channels();
        const Dtype* beta = this->blobs_[0]->gpu_data();
        const int div_factor = channel_shared_ ? channels : 1;
        
        // For in-place computation
        if (top[0] == bottom[0]) {
            caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
        }
        
        // NOLINT_NEXT_LINE(whitespace/operators)
        BestForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                                                                                count, channels, dim, bottom_data, top_data, beta, div_factor);
        CUDA_POST_KERNEL_CHECK;
    }
    
    template <typename Dtype>
    void BestLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* top_diff = top[0]->gpu_diff();
        const int count = bottom[0]->count();
        const int dim = bottom[0]->count(2);
        const int channels = bottom[0]->channels();
        const Dtype* beta = this->blobs_[0]->gpu_data();
        const int div_factor = channel_shared_ ? channels : 1;
        
        // For in-place computation
        if (top[0] == bottom[0]) {
            bottom_data = bottom_memory_.gpu_data();
        }
        
        // Propagate to param
        // Since to write bottom diff will affect top diff if top and bottom blobs
        // are identical (in-place computaion), we first compute param backward to
        // keep top_diff unchanged.
        if (this->param_propagate_down_[0]) {
            Dtype* beta_diff = this->blobs_[0]->mutable_gpu_diff();
            int cdim = channels * dim;
            Dtype dsum = 0.;
            for (int n = 0; n < bottom[0]->num(); ++n) {
                // compute element-wise diff
                // NOLINT_NEXT_LINE(whitespace/operators)
                BestParamBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
                CAFFE_CUDA_NUM_THREADS>>>(
                                          cdim, channels, dim, top_diff + top[0]->offset(n),
                                          bottom_data + bottom[0]->offset(n),
                                          backward_buff_.mutable_gpu_diff(),
                                          beta, div_factor);
                CUDA_POST_KERNEL_CHECK;
                if (channel_shared_) {
                    Dtype d;
                    caffe_gpu_dot<Dtype>(channels * dim, backward_buff_.gpu_diff(),
                                         multiplier_.gpu_data(), &d);
                    dsum += d;
                } else {
                    caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
                                          backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
                                          beta_diff);
                }
            }
            if (channel_shared_) {
                caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(dsum), beta_diff);
            }
        }
        // Propagate to bottom
        if (propagate_down[0]) {
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            const Dtype* beta = this->blobs_[0]->gpu_data();
            int div_factor = channel_shared_ ? channels : 1;
            // NOLINT_NEXT_LINE(whitespace/operators)
            BestBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS>>>(
                                      count, channels, dim, top_diff, bottom_data, bottom_diff, beta,
                                      div_factor);
            CUDA_POST_KERNEL_CHECK;
        }
    }
    
    
    INSTANTIATE_LAYER_GPU_FUNCS(BestLayer);
    
    
}  // namespace caffe
