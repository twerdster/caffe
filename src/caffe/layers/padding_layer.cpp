#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PaddingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PaddingParameter pad_param = this->layer_param_.padding_param();
  
  pad_l_ = pad_param.pad_l();
  pad_r_ = pad_param.pad_r();
  pad_t_ = pad_param.pad_t();
  pad_b_ = pad_param.pad_b();
  pad_value_ = pad_param.pad_value();
  
  CHECK_GT(pad_l_, 0) << "Padding ammount must be non negative.";
  CHECK_GT(pad_r_, 0) << "Padding ammount must be non negative.";
  CHECK_GT(pad_t_, 0) << "Padding ammount must be non negative.";
  CHECK_GT(pad_b_, 0) << "Padding ammount must be non negative.";

  CHECK(this->layer_param_.padding_param().pad_method()
      == PaddingParameter_PadMethod_CONSTANT
      || this->layer_param_.padding_param().pad_method()
      == PaddingParameter_PadMethod_REPEAT)
    << "Padding implemented only for CONSTANT and REPEAT. WRAP not yet implemented.";
}

template <typename Dtype>
void PaddingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ =   bottom[0]->height();
  width_ =    bottom[0]->width();

  padded_height_ = static_cast<int>(ceil(static_cast<float>( height_ + pad_t_ + pad_b_ ) ) );
  padded_width_  = static_cast<int>(ceil(static_cast<float>( width_  + pad_l_ + pad_r_ ) ) );

  top[0]->Reshape(bottom[0]->num(), channels_, padded_height_, padded_width_);
}

template <typename Dtype>
void PaddingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  // Different padding methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.padding_param().pad_method()) {
  case PaddingParameter_PadMethod_REPEAT:
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < padded_height_; ++ph) {
          for (int pw = 0; pw < padded_width_; ++pw) {
            const int padded_index = ph * padded_width_ + pw;
            int index = 0;

            int boxIdx = ((ph<pad_t_)?0:(ph<pad_t_+height_)?1:2)*3 + ((pw<pad_l_)?0:(pw<pad_l_+width_)?1:2);
            switch (boxIdx){
              case 0: index =           (0) * width_ +           (0); break; //Top Left 
              case 1: index =           (0) * width_ + (pw - pad_l_); break; //Top Center
              case 2: index =           (0) * width_ +  (width_ - 1); break; //Top Right
              case 3: index = (ph - pad_t_) * width_ +           (0); break; //Center Left
              case 4: index = (ph - pad_t_) * width_ + (pw - pad_l_); break; //Center Center
              case 5: index = (ph - pad_t_) * width_ +  (width_ - 1); break; //Center Right
              case 6: index = (height_ - 1) * width_ +           (0); break; //Bottom Left
              case 7: index = (height_ - 1) * width_ + (pw - pad_l_); break; //Bottom Center
              case 8: index = (height_ - 1) * width_ +  (width_ - 1); break; //Bottom Right
            }
            
            top_data[padded_index] = bottom_data[index];
          }
        }

        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case PaddingParameter_PadMethod_CONSTANT:
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < padded_height_; ++ph) {
          for (int pw = 0; pw < padded_width_; ++pw) {
            const int padded_index = ph * padded_width_ + pw;
            int index = 0;

            int boxIdx = ((ph<pad_t_)?0:(ph<pad_t_+height_)?1:2)*3 + ((pw<pad_l_)?0:(pw<pad_l_+width_)?1:2);
            switch (boxIdx){
              //case 0: index =           (0) * width_ +           (0); break; //Top Left 
              //case 1: index =           (0) * width_ + (pw - pad_l_); break; //Top Center
              //case 2: index =           (0) * width_ +  (width_ - 1); break; //Top Right
              //case 3: index = (ph - pad_t_) * width_ +           (0); break; //Center Left
              case 4: index = (ph - pad_t_) * width_ + (pw - pad_l_);          //Center Center
                      top_data[padded_index] = bottom_data[index];        break; 
              //case 5: index = (ph - pad_t_) * width_ +  (width_ - 1); break; //Center Right
              //case 6: index = (height_ - 1) * width_ +           (0); break; //Bottom Left
              //case 7: index = (height_ - 1) * width_ + (pw - pad_l_); break; //Bottom Center
              //case 8: index = (height_ - 1) * width_ +  (width_ - 1); break; //Bottom Right
              default: top_data[padded_index] = pad_value_;               break; 
            }
          }
        }

        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case PaddingParameter_PadMethod_WRAP:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown padding method.";
  }
}

template <typename Dtype>
void PaddingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different padding methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  switch (this->layer_param_.padding_param().pad_method()) {
  case PaddingParameter_PadMethod_REPEAT:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < padded_height_; ++ph) {
          for (int pw = 0; pw < padded_width_; ++pw) {
            const int padded_index = ph * padded_width_ + pw;
            int bottom_index = 0;

            int boxIdx = ((ph<pad_t_)?0:(ph<pad_t_+height_)?1:2)*3 + ((pw<pad_l_)?0:(pw<pad_l_+width_)?1:2);
            switch (boxIdx){
              case 0: bottom_index =           (0) * width_ +           (0); break; //Top Left 
              case 1: bottom_index =           (0) * width_ + (pw - pad_l_); break; //Top Center
              case 2: bottom_index =           (0) * width_ +  (width_ - 1); break; //Top Right
              case 3: bottom_index = (ph - pad_t_) * width_ +           (0); break; //Center Left
              case 4: bottom_index = (ph - pad_t_) * width_ + (pw - pad_l_); break; //Center Center
              case 5: bottom_index = (ph - pad_t_) * width_ +  (width_ - 1); break; //Center Right
              case 6: bottom_index = (height_ - 1) * width_ +           (0); break; //Bottom Left
              case 7: bottom_index = (height_ - 1) * width_ + (pw - pad_l_); break; //Bottom Center
              case 8: bottom_index = (height_ - 1) * width_ +  (width_ - 1); break; //Bottom Right
            }

            bottom_diff[bottom_index] += top_diff[padded_index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PaddingParameter_PadMethod_CONSTANT:
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < padded_height_; ++ph) {
          for (int pw = 0; pw < padded_width_; ++pw) {
            const int padded_index = ph * padded_width_ + pw;
            int bottom_index = 0;

            int boxIdx = ((ph<pad_t_)?0:(ph<pad_t_+height_)?1:2)*3 + ((pw<pad_l_)?0:(pw<pad_l_+width_)?1:2);
            switch (boxIdx){
              //case 0: bottom_index =           (0) * width_ +           (0); break; //Top Left 
              //case 1: bottom_index =           (0) * width_ + (pw - pad_l_); break; //Top Center
              //case 2: bottom_index =           (0) * width_ +  (width_ - 1); break; //Top Right
              //case 3: bottom_index = (ph - pad_t_) * width_ +           (0); break; //Center Left
              case 4: bottom_index = (ph - pad_t_) * width_ + (pw - pad_l_);          //Center Center
                      bottom_diff[bottom_index] += top_diff[padded_index];        break; 
              //case 5: bottom_index = (ph - pad_t_) * width_ +  (width_ - 1); break; //Center Right
              //case 6: bottom_index = (height_ - 1) * width_ +           (0); break; //Bottom Left
              //case 7: bottom_index = (height_ - 1) * width_ + (pw - pad_l_); break; //Bottom Center
              //case 8: bottom_index = (height_ - 1) * width_ +  (width_ - 1); break; //Bottom Right
              default: break; // I.e. derivative is zero and has no influence
            }
          }
        }

        // compute offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PaddingParameter_PadMethod_WRAP:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown padding method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(PaddingLayer);
#endif

INSTANTIATE_CLASS(PaddingLayer);

}  // namespace caffe
