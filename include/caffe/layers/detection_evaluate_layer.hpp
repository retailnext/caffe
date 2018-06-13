#ifndef CAFFE_DETECTION_EVALUATE_LAYER_HPP_
#define CAFFE_DETECTION_EVALUATE_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Generate the detection evaluation based on DetectionOutputLayer and
 * ground truth bounding box labels.
 *
 * Intended for use with MultiBox detection method.
 *
 * NOTE: does not implement Backwards operation.
 */
template <typename Dtype>
class DetectionEvaluateLayer : public Layer<Dtype> {
 public:
  explicit DetectionEvaluateLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DetectionEvaluate"; }
  virtual inline int ExactBottomBlobs() const { return 2; } //change 3 into 2 on March 10th 2017
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  /**
   * @brief Evaluate the detection output.
   *
   * @param bottom input Blob vector (exact 2)
   *   -# @f$ (1 \times 1 \times N \times 7) @f$
   *      N detection results.
   *   -# @f$ (1 \times 1 \times M \times 7) @f$
   *      M ground truth.
   * @param top Blob vector (length 1)
   *   -# @f$ (1 \times 1 \times N \times 4) @f$
   *      N is the number of detections, and each row is:
   *      [image_id, label, confidence, true_pos, false_pos]
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  int num_classes_;
  int background_label_id_;
  float overlap_threshold_;
  bool evaluate_difficult_gt_;
  vector<pair<int, int> > sizes_;
  int count_;
  //int gender_count_;
  int orientation_count_;
  bool use_normalized_bbox_;

  int num_orientation_classes_; //Added by Dong Liu for MTL
  int orientation_background_label_id_; //Added by Dong Liu for MTL
  //int num_gender_classes_; //Added by Dong Liu for MTL
  //int gender_background_label_id_; //Added by Dong Liu for MTL

};

}  // namespace caffe

#endif  // CAFFE_DETECTION_EVALUATE_LAYER_HPP_
