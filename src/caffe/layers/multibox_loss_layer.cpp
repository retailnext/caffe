#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/multibox_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(true); // location prediction
    this->layer_param_.add_propagate_down(true); // confidence prediction
    this->layer_param_.add_propagate_down(true); // orientation confidence prediction Added by Dong Liu for MTL
    this->layer_param_.add_propagate_down(false);
    this->layer_param_.add_propagate_down(false);
  }
  const MultiBoxLossParameter& multibox_loss_param =
      this->layer_param_.multibox_loss_param();

  num_ = bottom[0]->num();
  // num_priors_ = bottom[2]->height() / 4; commented by Dong Liu for MTL
  num_priors_ = bottom[3]->height()/4;  //Dong Liu for MTL
  // Get other parameters.
  CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
  num_classes_ = multibox_loss_param.num_classes();
  CHECK_GE(num_classes_, 1) << "num_classes should not be less than 1.";
  num_orientation_classes_ = multibox_loss_param.num_orientation_classes(); //added by Dong Liu for MTL
  CHECK_GE(num_orientation_classes_, 1)<<"num_orientation_classes should not be less than 1. "; //Added by Dong Liu for MTL
  share_location_ = multibox_loss_param.share_location();
  loc_classes_ = share_location_ ? 1 : num_classes_;
  match_type_ = multibox_loss_param.match_type();
  overlap_threshold_ = multibox_loss_param.overlap_threshold();
  use_prior_for_matching_ = multibox_loss_param.use_prior_for_matching();
  background_label_id_ = multibox_loss_param.background_label_id();
  orientation_background_label_id_ = multibox_loss_param.orientation_background_label_id(); //added by Dong Liu for MTL
    
  use_difficult_gt_ = multibox_loss_param.use_difficult_gt();
  do_neg_mining_ = multibox_loss_param.do_neg_mining();
  neg_pos_ratio_ = multibox_loss_param.neg_pos_ratio();
  neg_overlap_ = multibox_loss_param.neg_overlap();
  code_type_ = multibox_loss_param.code_type();
  encode_variance_in_target_ = multibox_loss_param.encode_variance_in_target();
  map_object_to_agnostic_ = multibox_loss_param.map_object_to_agnostic();
  if (map_object_to_agnostic_) {
    if (background_label_id_ >= 0) {
      CHECK_EQ(num_classes_, 2);
    } else {
      CHECK_EQ(num_classes_, 1);
    }
  }

  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }

  if (do_neg_mining_) {
    CHECK(share_location_)
        << "Currently only support negative mining if share_location is true.";
    CHECK_GT(neg_pos_ratio_, 0);
  }

  vector<int> loss_shape(1, 1);
  // Set up localization loss layer.
  loc_weight_ = multibox_loss_param.loc_weight();
  loc_loss_type_ = multibox_loss_param.loc_loss_type();
  // fake shape.
  vector<int> loc_shape(1, 1);
  loc_shape.push_back(4);
  loc_pred_.Reshape(loc_shape);
  loc_gt_.Reshape(loc_shape);
  loc_bottom_vec_.push_back(&loc_pred_);
  loc_bottom_vec_.push_back(&loc_gt_);
  loc_loss_.Reshape(loss_shape);
  loc_top_vec_.push_back(&loc_loss_);
  if (loc_loss_type_ == MultiBoxLossParameter_LocLossType_L2) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_l2_loc");
    layer_param.set_type("EuclideanLoss");
    layer_param.add_loss_weight(loc_weight_);
    loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
  } else if (loc_loss_type_ == MultiBoxLossParameter_LocLossType_SMOOTH_L1) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_smooth_L1_loc");
    layer_param.set_type("SmoothL1Loss");
    layer_param.add_loss_weight(loc_weight_);
    loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
  } else {
    LOG(FATAL) << "Unknown localization loss type.";
  }
  // Set up confidence loss layer.
  conf_loss_type_ = multibox_loss_param.conf_loss_type();
  conf_weight_ = multibox_loss_param.conf_weight();
  conf_bottom_vec_.push_back(&conf_pred_);
  conf_bottom_vec_.push_back(&conf_gt_);
  conf_loss_.Reshape(loss_shape);
  conf_top_vec_.push_back(&conf_loss_);
  if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_softmax_conf");
    layer_param.set_type("SoftmaxWithLoss");
    //layer_param.add_loss_weight(Dtype(1.));
    layer_param.add_loss_weight(conf_weight_);
    layer_param.mutable_loss_param()->set_normalization(
        LossParameter_NormalizationMode_NONE);
    SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
    softmax_param->set_axis(1);
    // Fake reshape.
    vector<int> conf_shape(1, 1);
    conf_gt_.Reshape(conf_shape);
    conf_shape.push_back(num_classes_);
    conf_pred_.Reshape(conf_shape);
    conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
  } else if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_logistic_conf");
    layer_param.set_type("SigmoidCrossEntropyLoss");
    layer_param.add_loss_weight(Dtype(1.));
    // Fake reshape.
    vector<int> conf_shape(1, 1);
    conf_shape.push_back(num_classes_);
    conf_gt_.Reshape(conf_shape);
    conf_pred_.Reshape(conf_shape);
    conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }
    
  // Set up orientation confidence loss layer
  orientation_weight_ = multibox_loss_param.orientation_weight();
  orientation_conf_loss_type_ = multibox_loss_param.conf_loss_type();
  orientation_conf_bottom_vec_.push_back(&orientation_conf_pred_);
  orientation_conf_bottom_vec_.push_back(&orientation_conf_gt_);
  orientation_conf_loss_.Reshape(loss_shape);
  orientation_conf_top_vec_.push_back(&orientation_conf_loss_);
  if (orientation_conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_orientation_softmax_conf");
        layer_param.set_type("SoftmaxWithLoss");
        layer_param.add_loss_weight(orientation_weight_);
        layer_param.mutable_loss_param()->set_normalization(
                                                            LossParameter_NormalizationMode_NONE);
        SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
        softmax_param->set_axis(1);
        // Fake reshape.
        vector<int> orientation_conf_shape(1, 1);
        orientation_conf_gt_.Reshape(orientation_conf_shape);
        orientation_conf_shape.push_back(num_orientation_classes_);
        orientation_conf_pred_.Reshape(orientation_conf_shape);
        orientation_conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        orientation_conf_loss_layer_->SetUp(orientation_conf_bottom_vec_, orientation_conf_top_vec_);
  } else if (orientation_conf_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
        LayerParameter layer_param;
        layer_param.set_name(this->layer_param_.name() + "_logistic_conf");
        layer_param.set_type("SigmoidCrossEntropyLoss");
        layer_param.add_loss_weight(Dtype(1.));
        // Fake reshape.
        vector<int> orientation_conf_shape(1, 1);
        orientation_conf_shape.push_back(num_orientation_classes_);
        orientation_conf_gt_.Reshape(orientation_conf_shape);
        orientation_conf_pred_.Reshape(orientation_conf_shape);
        orientation_conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
        orientation_conf_loss_layer_->SetUp(orientation_conf_bottom_vec_, orientation_conf_top_vec_);
  } else {
      LOG(FATAL) << "Unknown confidence loss type.";
  }
    
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  num_ = bottom[0]->num();
  // num_priors_ = bottom[2]->height() / 4; commnented by DOng Liu for MTL
  num_priors_ = bottom[3]->height() / 4; //Dong Liu for MTL
  // num_gt_ = bottom[3]->height(); commented by Dong Liu for MTL
  num_gt_ = bottom[4]->height(); //Dong Liu for MTL
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num()); //Dong Liu for MTL
  CHECK_EQ(num_priors_ * loc_classes_ * 4, bottom[0]->channels())
      << "Number of priors must match number of location predictions.";
  CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
      << "Number of priors must match number of confidence predictions.";
  CHECK_EQ(num_priors_ * num_orientation_classes_, bottom[2]->channels()) //Dong Liu for MTL
    << "Number of priors must match number of orientation confidence predictions.";
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->cpu_data();
  const Dtype* conf_data = bottom[1]->cpu_data();
  const Dtype* orientation_conf_data = bottom[2]->cpu_data();
  const Dtype* prior_data = bottom[3]->cpu_data();
  const Dtype* gt_data = bottom[4]->cpu_data();

  // Retrieve all ground truth.
  map<int, vector<NormalizedBBox> > all_gt_bboxes;
  GetGroundTruth(gt_data, num_gt_, background_label_id_, use_difficult_gt_,
                 &all_gt_bboxes);

  // Retrieve all orientation ground truth
  map<int, vector<NormalizedBBox> > all_orientation_gt_bboxes;
  GetOrientationGroundTruth(gt_data, num_gt_, orientation_background_label_id_, use_difficult_gt_,
                       &all_orientation_gt_bboxes); //TODO: Need to add this function for orientations


  // Retrieve all prior bboxes. It is same within a batch since we assume all
  // images in a batch are of same dimension.
  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float> > prior_variances;
  GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

  // Retrieve all predictions.
  vector<LabelBBox> all_loc_preds;
  GetLocPredictions(loc_data, num_, num_priors_, loc_classes_, share_location_,
                    &all_loc_preds);

  // Retrieve max scores for each prior. Used in negative mining.
  vector<vector<float> > all_max_scores;
  vector<vector<float> > all_orientation_max_scores; //added on April 6th 2017
  if (do_neg_mining_) {
    GetMaxConfidenceScores(conf_data, num_, num_priors_, num_classes_,
                           background_label_id_, conf_loss_type_,
                           &all_max_scores);
	GetMaxConfidenceScores(orientation_conf_data, num_, num_priors_, num_orientation_classes_,
		                   orientation_background_label_id_, conf_loss_type_,
		                   &all_orientation_max_scores);
  }

  num_matches_ = 0;
  int num_negs = 0;
  for (int i = 0; i < num_; ++i) {
    map<int, vector<int> > match_indices;
    vector<int> neg_indices;
    // Check if there is ground truth for current image.
    if (all_gt_bboxes.find(i) == all_gt_bboxes.end()) {
      // There is no gt for current image. All predictions are negative.
      all_match_indices_.push_back(match_indices);
      all_neg_indices_.push_back(neg_indices);
      continue;
    }
    // Find match between predictions and ground truth.
    const vector<NormalizedBBox>& gt_bboxes = all_gt_bboxes.find(i)->second;
    map<int, vector<float> > match_overlaps;
    if (!use_prior_for_matching_) {
      for (int c = 0; c < loc_classes_; ++c) {
        int label = share_location_ ? -1 : c;
        if (!share_location_ && label == background_label_id_) {
          // Ignore background loc predictions.
          continue;
        }
        // Decode the prediction into bbox first.
        vector<NormalizedBBox> loc_bboxes;
        DecodeBBoxes(prior_bboxes, prior_variances,
                     code_type_, encode_variance_in_target_,
                     all_loc_preds[i][label], &loc_bboxes);
        MatchBBox(gt_bboxes, loc_bboxes, label, match_type_,
                  overlap_threshold_, &match_indices[label],
                  &match_overlaps[label]);
      }
    } else {
      // Use prior bboxes to match against all ground truth.
      vector<int> temp_match_indices;
      vector<float> temp_match_overlaps;
      const int label = -1;
      MatchBBox(gt_bboxes, prior_bboxes, label, match_type_, overlap_threshold_,
                &temp_match_indices, &temp_match_overlaps);
      if (share_location_) {
        match_indices[label] = temp_match_indices;
        match_overlaps[label] = temp_match_overlaps;
      } else {
        // Get ground truth label for each ground truth bbox.
        vector<int> gt_labels;
        for (int g = 0; g < gt_bboxes.size(); ++g) {
          gt_labels.push_back(gt_bboxes[g].label());
        }
        // Distribute the matching results to different loc_class.
        for (int c = 0; c < loc_classes_; ++c) {
          if (c == background_label_id_) {
            // Ignore background loc predictions.
            continue;
          }
          match_indices[c].resize(temp_match_indices.size(), -1);
          match_overlaps[c] = temp_match_overlaps;
          for (int m = 0; m < temp_match_indices.size(); ++m) {
            if (temp_match_indices[m] != -1) {
              const int gt_idx = temp_match_indices[m];
              CHECK_LT(gt_idx, gt_labels.size());
              if (c == gt_labels[gt_idx]) {
                match_indices[c][m] = gt_idx;
              }
            }
          }
        }
      }
    }
    // Record matching statistics.
    for (map<int, vector<int> >::iterator it = match_indices.begin();
         it != match_indices.end(); ++it) {
      const int label = it->first;
      // Get positive indices.
      int num_pos = 0;
      for (int m = 0; m < match_indices[label].size(); ++m) {
        if (match_indices[label][m] != -1) {
          ++num_pos;
        }
      }
      num_matches_ += num_pos;
      if (do_neg_mining_) {
        // Get max scores for all the non-matched priors.
        vector<pair<float, int> > scores_indices;
        int num_neg = 0;
        for (int m = 0; m < match_indices[label].size(); ++m) {
          if (match_indices[label][m] == -1 &&
              match_overlaps[label][m] < neg_overlap_) {
            scores_indices.push_back(std::make_pair(all_max_scores[i][m], m));
            ++num_neg;
          }
        }
        // Pick top num_neg negatives.
        num_neg = std::min(static_cast<int>(num_pos * neg_pos_ratio_), num_neg);
        std::sort(scores_indices.begin(), scores_indices.end(),
                  SortScorePairDescend<int>);
        for (int n = 0; n < num_neg; ++n) {
          neg_indices.push_back(scores_indices[n].second);
        }
        num_negs += num_neg;
      }
    }
    all_match_indices_.push_back(match_indices);
    all_neg_indices_.push_back(neg_indices);
  }

  if (num_matches_ >= 1) {
    // Form data to pass on to loc_loss_layer_.
    vector<int> loc_shape(2);
    loc_shape[0] = 1;
    loc_shape[1] = num_matches_ * 4;
    loc_pred_.Reshape(loc_shape);
    loc_gt_.Reshape(loc_shape);
    Dtype* loc_pred_data = loc_pred_.mutable_cpu_data();
    Dtype* loc_gt_data = loc_gt_.mutable_cpu_data();
    int count = 0;
    for (int i = 0; i < num_; ++i) {
      for (map<int, vector<int> >::iterator it = all_match_indices_[i].begin();
           it != all_match_indices_[i].end(); ++it) {
        const int label = it->first;
        const vector<int>& match_index = it->second;
        CHECK(all_loc_preds[i].find(label) != all_loc_preds[i].end());
        const vector<NormalizedBBox>& loc_pred = all_loc_preds[i][label];
        for (int j = 0; j < match_index.size(); ++j) {
          if (match_index[j] == -1) {
            continue;
          }
          // Store location prediction.
          CHECK_LT(j, loc_pred.size());
          loc_pred_data[count * 4] = loc_pred[j].xmin();
          loc_pred_data[count * 4 + 1] = loc_pred[j].ymin();
          loc_pred_data[count * 4 + 2] = loc_pred[j].xmax();
          loc_pred_data[count * 4 + 3] = loc_pred[j].ymax();
          // Store encoded ground truth.
          const int gt_idx = match_index[j];
          CHECK(all_gt_bboxes.find(i) != all_gt_bboxes.end());
          CHECK_LT(gt_idx, all_gt_bboxes[i].size());
          const NormalizedBBox& gt_bbox = all_gt_bboxes[i][gt_idx];
          NormalizedBBox gt_encode;
          CHECK_LT(j, prior_bboxes.size());
          EncodeBBox(prior_bboxes[j], prior_variances[j], code_type_,
                     encode_variance_in_target_, gt_bbox, &gt_encode);
          loc_gt_data[count * 4] = gt_encode.xmin();
          loc_gt_data[count * 4 + 1] = gt_encode.ymin();
          loc_gt_data[count * 4 + 2] = gt_encode.xmax();
          loc_gt_data[count * 4 + 3] = gt_encode.ymax();
          if (encode_variance_in_target_) {
            for (int k = 0; k < 4; ++k) {
              CHECK_GT(prior_variances[j][k], 0);
              loc_pred_data[count * 4 + k] /= prior_variances[j][k];
              loc_gt_data[count * 4 + k] /= prior_variances[j][k];
            }
          }
          ++count;
        }
      }
    }
    loc_loss_layer_->Reshape(loc_bottom_vec_, loc_top_vec_);
    loc_loss_layer_->Forward(loc_bottom_vec_, loc_top_vec_);
  } else {
    loc_loss_.mutable_cpu_data()[0] = 0;
  }

  // Form data to pass on to conf_loss_layer_, age_conf_loss_layer, gender_conf_loss_layer Modified by Dong Liu for MTL.
  if (do_neg_mining_) {
    num_conf_ = num_matches_ + num_negs;
  } else {
    num_conf_ = num_ * num_priors_;
  }
  if (num_conf_ >= 1) {
    // Reshape the confidence data.
    vector<int> conf_shape;
    if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
      conf_shape.push_back(num_conf_);
      conf_gt_.Reshape(conf_shape);
      conf_shape.push_back(num_classes_);
      conf_pred_.Reshape(conf_shape);
    } else if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
      conf_shape.push_back(1);
      conf_shape.push_back(num_conf_);
      conf_shape.push_back(num_classes_);
      conf_gt_.Reshape(conf_shape);
      conf_pred_.Reshape(conf_shape);
    } else {
      LOG(FATAL) << "Unknown confidence loss type.";
    }
    if (!do_neg_mining_) {
      // Consider all scores.
      // Share data and diff with bottom[1].
      CHECK_EQ(conf_pred_.count(), bottom[1]->count());
      conf_pred_.ShareData(*(bottom[1]));
    }
    Dtype* conf_pred_data = conf_pred_.mutable_cpu_data();
    Dtype* conf_gt_data = conf_gt_.mutable_cpu_data();
    caffe_set(conf_gt_.count(), Dtype(background_label_id_), conf_gt_data);
    int count = 0;
    for (int i = 0; i < num_; ++i) {
      if (all_gt_bboxes.find(i) != all_gt_bboxes.end()) {
        // Save matched (positive) bboxes scores and labels.
        const map<int, vector<int> >& match_indices = all_match_indices_[i];
        for (int j = 0; j < num_priors_; ++j) {
          for (map<int, vector<int> >::const_iterator it =
               match_indices.begin(); it != match_indices.end(); ++it) {
            const vector<int>& match_index = it->second;
            CHECK_EQ(match_index.size(), num_priors_);
            if (match_index[j] == -1) {
              continue;
            }
            const int gt_label = map_object_to_agnostic_ ?
                background_label_id_ + 1 :
                all_gt_bboxes[i][match_index[j]].label();
            int idx = do_neg_mining_ ? count : j;
            switch (conf_loss_type_) {
              case MultiBoxLossParameter_ConfLossType_SOFTMAX:
                conf_gt_data[idx] = gt_label;
                break;
              case MultiBoxLossParameter_ConfLossType_LOGISTIC:
                conf_gt_data[idx * num_classes_ + gt_label] = 1;
                break;
              default:
                LOG(FATAL) << "Unknown conf loss type.";
            }
            if (do_neg_mining_) {
              // Copy scores for matched bboxes.
              caffe_copy<Dtype>(num_classes_, conf_data + j * num_classes_,
                                conf_pred_data + count * num_classes_);
              ++count;
            }
          }
        }
        if (do_neg_mining_) {
          // Save negative bboxes scores and labels.
          for (int n = 0; n < all_neg_indices_[i].size(); ++n) {
            int j = all_neg_indices_[i][n];
            CHECK_LT(j, num_priors_);
            caffe_copy<Dtype>(num_classes_, conf_data + j * num_classes_,
                              conf_pred_data + count * num_classes_);
            switch (conf_loss_type_) {
              case MultiBoxLossParameter_ConfLossType_SOFTMAX:
                conf_gt_data[count] = background_label_id_;
                break;
              case MultiBoxLossParameter_ConfLossType_LOGISTIC:
                conf_gt_data[count * num_classes_ + background_label_id_] = 1;
                break;
              default:
                LOG(FATAL) << "Unknown conf loss type.";
            }
            ++count;
          }
        }
      }
      // Go to next image.
      if (do_neg_mining_) {
        conf_data += bottom[1]->offset(1);
      } else {
        conf_gt_data += num_priors_;
      }
    }
    conf_loss_layer_->Reshape(conf_bottom_vec_, conf_top_vec_);
    conf_loss_layer_->Forward(conf_bottom_vec_, conf_top_vec_);
  } else {
    conf_loss_.mutable_cpu_data()[0] = 0;
  }

  //Added on April 6th 2017
  // Below orientation negative mining, added by Dong Liu for MTL
  orientation_num_matches_ = 0;
  int orientation_num_negs = 0;
  for (int i = 0; i < num_; ++i) {
    map<int, vector<int> > orientation_match_indices;
    vector<int> orientation_neg_indices;
      // Check if there is ground truth for current image.
     if (all_orientation_gt_bboxes.find(i) == all_orientation_gt_bboxes.end()) {
        // There is no gt for current image. All predictions are negative.
        all_orientation_match_indices_.push_back(orientation_match_indices);
        all_orientation_neg_indices_.push_back(orientation_neg_indices);
        continue;
     }
      // Find match between predictions and ground truth.
      const vector<NormalizedBBox>& orientation_gt_bboxes = all_orientation_gt_bboxes.find(i)->second;
      map<int, vector<float> > orientation_match_overlaps;
      if (!use_prior_for_matching_) {
        for (int c = 0; c < loc_classes_; ++c) {
          int label = share_location_ ? -1 : c;
          if (!share_location_ && label == orientation_background_label_id_) {
            // Ignore background loc predictions.
            continue;
          }
          // Decode the prediction into bbox first.
          vector<NormalizedBBox> loc_bboxes;
          DecodeBBoxes(prior_bboxes, prior_variances,
                       code_type_, encode_variance_in_target_,
                       all_loc_preds[i][label], &loc_bboxes);
          MatchBBox(orientation_gt_bboxes, loc_bboxes, label, match_type_,
                    overlap_threshold_, &orientation_match_indices[label],
                    &orientation_match_overlaps[label]);
        }
      } else {
        // Use prior bboxes to match against all ground truth.
        vector<int> temp_orientation_match_indices;
        vector<float> temp_orientation_match_overlaps;
        const int label = -1;
        MatchBBox(orientation_gt_bboxes, prior_bboxes, label, match_type_, overlap_threshold_,
                  &temp_orientation_match_indices, &temp_orientation_match_overlaps);
        if (share_location_) {
          orientation_match_indices[label] = temp_orientation_match_indices;
          orientation_match_overlaps[label] = temp_orientation_match_overlaps;
        } else {
          // Get ground truth label for each ground truth bbox.
          vector<int> orientation_gt_labels;
          for (int g = 0; g < orientation_gt_bboxes.size(); ++g) {
            orientation_gt_labels.push_back(orientation_gt_bboxes[g].orientation());
          }
          // Distribute the matching results to different loc_class.
          for (int c = 0; c < loc_classes_; ++c) {
            if (c == orientation_background_label_id_) {
              // Ignore background loc predictions.
              continue;
            }
            orientation_match_indices[c].resize(temp_orientation_match_indices.size(), -1);
            orientation_match_overlaps[c] = temp_orientation_match_overlaps;
            for (int m = 0; m < temp_orientation_match_indices.size(); ++m) {
              if (temp_orientation_match_indices[m] != -1) {
                const int orientation_gt_idx = temp_orientation_match_indices[m];
                CHECK_LT(orientation_gt_idx, orientation_gt_labels.size());
                if (c == orientation_gt_labels[orientation_gt_idx]) {
                  orientation_match_indices[c][m] = orientation_gt_idx;
                }
              }
            }
          }
        }
      }
      // Record matching statistics.
      for (map<int, vector<int> >::iterator it = orientation_match_indices.begin();
           it != orientation_match_indices.end(); ++it) {
        const int label = it->first;
        // Get positive indices.
        int num_pos = 0;
        for (int m = 0; m < orientation_match_indices[label].size(); ++m) {
          if (orientation_match_indices[label][m] != -1) {
            ++num_pos;
          }
        }
        orientation_num_matches_ += num_pos;
        if (do_neg_mining_) {
          // Get max scores for all the non-matched priors.
          vector<pair<float, int> > orientation_scores_indices;
          int num_neg = 0;
          for (int m = 0; m < orientation_match_indices[label].size(); ++m) {
            if (orientation_match_indices[label][m] == -1 &&
                orientation_match_overlaps[label][m] < neg_overlap_) {
                orientation_scores_indices.push_back(std::make_pair(all_orientation_max_scores[i][m], m));
                ++num_neg;
            }
          }
          // Pick top num_neg negatives.
          num_neg = std::min(static_cast<int>(num_pos * neg_pos_ratio_), num_neg);
          std::sort(orientation_scores_indices.begin(), orientation_scores_indices.end(),
                    SortScorePairDescend<int>);
          for (int n = 0; n < num_neg; ++n) {
            orientation_neg_indices.push_back(orientation_scores_indices[n].second);
          }
          orientation_num_negs += num_neg;
        }
      }
      all_orientation_match_indices_.push_back(orientation_match_indices);
      all_orientation_neg_indices_.push_back(orientation_neg_indices);
    }

    // Form data to pass on to gender_conf_loss_layer Modified by Dong Liu for MTL.
    if (do_neg_mining_) {
      orientation_num_conf_ = orientation_num_matches_ + orientation_num_negs;
    } else {
      orientation_num_conf_ = num_ * num_priors_;
    }
    if (orientation_num_conf_ >= 1) {

        //Reshape the orientation confidence data.
        vector<int> orientation_conf_shape;
        if (orientation_conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
        	orientation_conf_shape.push_back(orientation_num_conf_);
        	orientation_conf_gt_.Reshape(orientation_conf_shape);
        	orientation_conf_shape.push_back(num_orientation_classes_);
        	orientation_conf_pred_.Reshape(orientation_conf_shape);
        } else if (orientation_conf_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
        	orientation_conf_shape.push_back(1);
        	orientation_conf_shape.push_back(orientation_num_conf_);
        	orientation_conf_shape.push_back(num_orientation_classes_);
        	orientation_conf_gt_.Reshape(orientation_conf_shape);
        	orientation_conf_pred_.Reshape(orientation_conf_shape);
        } else {
            LOG(FATAL) << "Unknown orientation confidence loss type.";
        }

      if (!do_neg_mining_) {
        // Consider all scores.
        // Share data and diff with bottom[1].
        CHECK_EQ(orientation_conf_pred_.count(), bottom[2]->count());
        orientation_conf_pred_.ShareData(*(bottom[2]));
      }
      Dtype* orientation_conf_pred_data = orientation_conf_pred_.mutable_cpu_data();
      Dtype* orientation_conf_gt_data = orientation_conf_gt_.mutable_cpu_data();
      caffe_set(orientation_conf_gt_.count(), Dtype(orientation_background_label_id_), orientation_conf_gt_data);

      int count = 0;
      for (int i = 0; i < num_; ++i) {
        if (all_orientation_gt_bboxes.find(i) != all_orientation_gt_bboxes.end()) {
          // Save matched (positive) bboxes scores and labels.
          const map<int, vector<int> >& orientation_match_indices = all_orientation_match_indices_[i];
          for (int j = 0; j < num_priors_; ++j) {
            for (map<int, vector<int> >::const_iterator it =
                 orientation_match_indices.begin(); it != orientation_match_indices.end(); ++it) {
              const vector<int>& orientation_match_index = it->second;
              CHECK_EQ(orientation_match_index.size(), num_priors_);
              if (orientation_match_index[j] == -1) {
                continue;
              }
              const int orientation_gt_label = map_object_to_agnostic_ ?
                  orientation_background_label_id_ + 1 :
                  all_orientation_gt_bboxes[i][orientation_match_index[j]].orientation();

              //const int age_gt_label = all_gt_bboxes[i][match_index[j]].age(); //Dong Liu for MTL.
              //const int gender_gt_label = all_gender_gt_bboxes[i][match_index[j]]; //Dong Liu for MTL

              int idx = do_neg_mining_ ? count : j;

              /*switch (conf_loss_type_) {
                case MultiBoxLossParameter_ConfLossType_SOFTMAX:
                  gender_conf_gt_data[idx] = gender_gt_label;
                  break;
                case MultiBoxLossParameter_ConfLossType_LOGISTIC:
                  gender_conf_gt_data[idx * num_gender_classes_ + gender_gt_label] = 1;
                  break;
                default:
                  LOG(FATAL) << "Unknown conf loss type.";
              }*/

              // Below added by Dong Liu for MTL
             /* switch (age_conf_loss_type_) {
                 case MultiBoxLossParameter_ConfLossType_SOFTMAX:
                        age_conf_gt_data[idx] = age_gt_label;
                        break;
                  case MultiBoxLossParameter_ConfLossType_LOGISTIC:
                        age_conf_gt_data[idx * num_age_classes_ + age_gt_label] = 1;
                        break;
                  default:
                        LOG(FATAL) << "Unknown age conf loss type.";
              }*/

                // Below added by Dong Liu for MTL
                switch (orientation_conf_loss_type_) {
                    case MultiBoxLossParameter_ConfLossType_SOFTMAX:
                        orientation_conf_gt_data[idx] = orientation_gt_label;
                        break;
                    case MultiBoxLossParameter_ConfLossType_LOGISTIC:
                        orientation_conf_gt_data[idx * num_orientation_classes_ + orientation_gt_label] = 1;
                        break;
                    default:
                        LOG(FATAL) << "Unknown orientation conf loss type.";
                }


              if (do_neg_mining_) {
                // Copy scores for matched bboxes.
                //caffe_copy<Dtype>(num_classes_, conf_data + j * num_classes_,
                 //                 conf_pred_data + count * num_classes_);
                //caffe_copy<Dtype>(num_age_classes_, age_conf_data + j * num_age_classes_,
                //                    age_conf_pred_data + count * num_age_classes_); //Added by Dong Liu for MTL
                caffe_copy<Dtype>(num_orientation_classes_, orientation_conf_data + j * num_orientation_classes_,
                                  orientation_conf_pred_data + count * num_orientation_classes_); //Added by Dong Liu for MTL

                ++count;
              }
            }
          }

          if (do_neg_mining_) {
            // Save negative bboxes scores and labels.
            for (int n = 0; n < all_orientation_neg_indices_[i].size(); ++n) {
              int j = all_orientation_neg_indices_[i][n];
              CHECK_LT(j, num_priors_);
              //caffe_copy<Dtype>(num_classes_, conf_data + j * num_classes_,
              //                  conf_pred_data + count * num_classes_);

              //caffe_copy<Dtype>(num_age_classes_, age_conf_data + j * num_age_classes_,
              //                    age_conf_pred_data + count * num_age_classes_); //added by Dong Liu for MTL

              caffe_copy<Dtype>(num_orientation_classes_, orientation_conf_data + j * num_orientation_classes_,
                                orientation_conf_pred_data + count * num_orientation_classes_); //added by Dong Liu for MTL


              /*switch (conf_loss_type_) {
                case MultiBoxLossParameter_ConfLossType_SOFTMAX:
                  conf_gt_data[count] = background_label_id_;
                  break;
                case MultiBoxLossParameter_ConfLossType_LOGISTIC:
                  conf_gt_data[count * num_classes_ + background_label_id_] = 1;
                  break;
                default:
                  LOG(FATAL) << "Unknown conf loss type.";
              }*/

              // Below added by Dong Liu for MTL
              /*  switch (age_conf_loss_type_) {
                    case MultiBoxLossParameter_ConfLossType_SOFTMAX:
                        age_conf_gt_data[count] = age_background_label_id_;
                        break;
                    case MultiBoxLossParameter_ConfLossType_LOGISTIC:
                        age_conf_gt_data[count * num_age_classes_ + age_background_label_id_] = 1;
                        break;
                    default:
                        LOG(FATAL) << "Unknown age conf loss type.";
                } */

                // Below added by Dong Liu for MTL
                switch (orientation_conf_loss_type_) {
                    case MultiBoxLossParameter_ConfLossType_SOFTMAX:
                        orientation_conf_gt_data[count] = orientation_background_label_id_;
                        break;
                    case MultiBoxLossParameter_ConfLossType_LOGISTIC:
                        orientation_conf_gt_data[count * num_orientation_classes_ + orientation_background_label_id_] = 1;
                        break;
                    default:
                        LOG(FATAL) << "Unknown orientation conf loss type.";
                }

              ++count;
            }
          }
        }
        // Go to next image.
        if (do_neg_mining_) {
          //conf_data += bottom[1]->offset(1);
          //age_conf_data += bottom[2]->offset(1);
          orientation_conf_data += bottom[2]->offset(1);

        } else {
          //conf_gt_data += num_priors_;
          //age_conf_gt_data += num_priors_;
          orientation_conf_gt_data += num_priors_;
        }
      }
      //conf_loss_layer_->Reshape(conf_bottom_vec_, conf_top_vec_);
      //conf_loss_layer_->Forward(conf_bottom_vec_, conf_top_vec_);

      //age_conf_loss_layer_->Reshape(age_conf_bottom_vec_, age_conf_top_vec_);
      //age_conf_loss_layer_->Forward(age_conf_bottom_vec_, age_conf_top_vec_);

      orientation_conf_loss_layer_->Reshape(orientation_conf_bottom_vec_, orientation_conf_top_vec_);
      orientation_conf_loss_layer_->Forward(orientation_conf_bottom_vec_, orientation_conf_top_vec_);

    }


  top[0]->mutable_cpu_data()[0] = 0;

  top[0]->mutable_cpu_data()[1] = 0; // loc_loss_, Added by dong Liu for MTL
  top[0]->mutable_cpu_data()[2] = 0; //loc normalizer, added by Dong Liu for MTL


  top[0]->mutable_cpu_data()[3] = 0; //conf_loss, added by Dong Liu for MTL
  top[0]->mutable_cpu_data()[4] = 0; //conf normalizer, added by Dong Liu for MTL


  top[0]->mutable_cpu_data()[5] = 0; //orientation_loss_. added by Dong Liu for MTL
  top[0]->mutable_cpu_data()[6] = 0; //Normalizer, added by Dong Liu for MTL


  if (this->layer_param_.propagate_down(0)) {
    // TODO(weiliu89): Understand why it needs to divide 2.
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, num_, num_priors_, num_matches_);
    top[0]->mutable_cpu_data()[0] +=
        loc_weight_ * loc_loss_.cpu_data()[0] / normalizer;
    top[0]->mutable_cpu_data()[1] = loc_loss_.cpu_data()[0];// Added by Dong Liu for MTL
    top[0]->mutable_cpu_data()[2] = normalizer;
  }
  if (this->layer_param_.propagate_down(1)) {
    // TODO(weiliu89): Understand why it needs to divide 2.
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, num_, num_priors_, num_matches_);
    top[0]->mutable_cpu_data()[0] += conf_weight_ * conf_loss_.cpu_data()[0] / normalizer;
    top[0]->mutable_cpu_data()[3] = conf_loss_.cpu_data()[0];// Added by Dong Liu for MTL
    top[0]->mutable_cpu_data()[4] = normalizer;
  }
  if (this->layer_param_.propagate_down(2)) {
    //Added by Dong Liu for MTL
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
           normalization_, num_, num_priors_, orientation_num_matches_);
    top[0]->mutable_cpu_data()[0] +=
    orientation_weight_ * orientation_conf_loss_.cpu_data()[0] / normalizer;
    top[0]->mutable_cpu_data()[5] = orientation_conf_loss_.cpu_data()[0];// Added by Dong Liu for MTL
    top[0]->mutable_cpu_data()[6] = normalizer;
  }
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  // Commented by Dong Liu for MTL
  /*
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to prior inputs.";
  }
  if (propagate_down[3]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }
  */
  // Added by Dong Liu for MTL
  if (propagate_down[3]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to prior inputs.";
  }
  if (propagate_down[4]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }

  // Back propagate on location prediction.
  if (propagate_down[0]) {
    Dtype* loc_bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), loc_bottom_diff);
    if (num_matches_ >= 1) {
      vector<bool> loc_propagate_down;
      // Only back propagate on prediction, not ground truth.
      loc_propagate_down.push_back(true);
      loc_propagate_down.push_back(false);
      loc_loss_layer_->Backward(loc_top_vec_, loc_propagate_down,
                                loc_bottom_vec_);
      // Scale gradient.
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_priors_, num_matches_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      caffe_scal(loc_pred_.count(), loss_weight, loc_pred_.mutable_cpu_diff());
      // Copy gradient back to bottom[0].
      const Dtype* loc_pred_diff = loc_pred_.cpu_diff();
      int count = 0;
      for (int i = 0; i < num_; ++i) {
        for (map<int, vector<int> >::iterator it =
             all_match_indices_[i].begin();
             it != all_match_indices_[i].end(); ++it) {
          const int label = share_location_ ? 0 : it->first;
          const vector<int>& match_index = it->second;
          for (int j = 0; j < match_index.size(); ++j) {
            if (match_index[j] == -1) {
              continue;
            }
            // Copy the diff to the right place.
            int start_idx = loc_classes_ * 4 * j + label * 4;
            caffe_copy<Dtype>(4, loc_pred_diff + count * 4,
                              loc_bottom_diff + start_idx);
            ++count;
          }
        }
        loc_bottom_diff += bottom[0]->offset(1);
      }
    }
  }

  // Back propagate on confidence prediction.
  if (propagate_down[1]) {
    Dtype* conf_bottom_diff = bottom[1]->mutable_cpu_diff();
    caffe_set(bottom[1]->count(), Dtype(0), conf_bottom_diff);
    if (num_conf_ >= 1) {
      vector<bool> conf_propagate_down;
      // Only back propagate on prediction, not ground truth.
      conf_propagate_down.push_back(true);
      conf_propagate_down.push_back(false);
      conf_loss_layer_->Backward(conf_top_vec_, conf_propagate_down,
                                 conf_bottom_vec_);
      // Scale gradient.
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_priors_, num_matches_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      caffe_scal(conf_pred_.count(), loss_weight,
                 conf_pred_.mutable_cpu_diff());
      // Copy gradient back to bottom[1].
      const Dtype* conf_pred_diff = conf_pred_.cpu_diff();
      if (do_neg_mining_) {
        int count = 0;
        for (int i = 0; i < num_; ++i) {
          // Copy matched (positive) bboxes scores' diff.
          const map<int, vector<int> >& match_indices = all_match_indices_[i];
          for (int j = 0; j < num_priors_; ++j) {
            for (map<int, vector<int> >::const_iterator it =
                 match_indices.begin(); it != match_indices.end(); ++it) {
              const vector<int>& match_index = it->second;
              CHECK_EQ(match_index.size(), num_priors_);
              if (match_index[j] == -1) {
                continue;
              }
              // Copy the diff to the right place.
              caffe_copy<Dtype>(num_classes_,
                                conf_pred_diff + count * num_classes_,
                                conf_bottom_diff + j * num_classes_);
              ++count;
            }
          }
          // Copy negative bboxes scores' diff.
          for (int n = 0; n < all_neg_indices_[i].size(); ++n) {
            int j = all_neg_indices_[i][n];
            CHECK_LT(j, num_priors_);
            caffe_copy<Dtype>(num_classes_,
                              conf_pred_diff + count * num_classes_,
                              conf_bottom_diff + j * num_classes_);
            ++count;
          }
          conf_bottom_diff += bottom[1]->offset(1);
        }
      } else {
        // The diff is already computed and stored.
        bottom[1]->ShareDiff(conf_pred_);
      }
    }
  }


    //added on April 6th 2017
    // Back propagate on orientation confidence prediction. Dong Liu for MTL
    if (propagate_down[2]) {
        Dtype* orientation_conf_bottom_diff = bottom[2]->mutable_cpu_diff();
        caffe_set(bottom[2]->count(), Dtype(0), orientation_conf_bottom_diff);
        if (orientation_num_conf_ >= 1) {
            vector<bool> orientation_conf_propagate_down;
            // Only back propagate on prediction, not ground truth.
            orientation_conf_propagate_down.push_back(true);
            orientation_conf_propagate_down.push_back(false);
            orientation_conf_loss_layer_->Backward(orientation_conf_top_vec_, orientation_conf_propagate_down,
                                                   orientation_conf_bottom_vec_);
            // Scale gradient.
            Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
                                                               normalization_, num_, num_priors_, orientation_num_matches_);
            Dtype orientation_loss_weight =  top[0]->cpu_diff()[0] / normalizer;
            caffe_scal(orientation_conf_pred_.count(), orientation_loss_weight,
                       orientation_conf_pred_.mutable_cpu_diff());
            // Copy gradient back to bottom[2].
            const Dtype* orientation_conf_pred_diff = orientation_conf_pred_.cpu_diff();
            if (do_neg_mining_) {
                int count = 0;
                for (int i = 0; i < num_; ++i) {
                    // Copy matched (positive) bboxes scores' diff.
                    const map<int, vector<int> >& orientation_match_indices = all_orientation_match_indices_[i];
                    for (int j = 0; j < num_priors_; ++j) {
                        for (map<int, vector<int> >::const_iterator it =
                            orientation_match_indices.begin(); it != orientation_match_indices.end(); ++it) {
                            const vector<int>& orientation_match_index = it->second;
                            CHECK_EQ(orientation_match_index.size(), num_priors_);
                            if (orientation_match_index[j] == -1) {
                                continue;
                            }
                            // Copy the diff to the right place.
                            caffe_copy<Dtype>(num_orientation_classes_,
                                              orientation_conf_pred_diff + count * num_orientation_classes_,
                                              orientation_conf_bottom_diff + j * num_orientation_classes_);
                            ++count;
                        }
                    }
                    // Copy negative bboxes scores' diff.
                    for (int n = 0; n < all_orientation_neg_indices_[i].size(); ++n) {
                        int j = all_orientation_neg_indices_[i][n];
                        CHECK_LT(j, num_priors_);
                        caffe_copy<Dtype>(num_orientation_classes_,
                                          orientation_conf_pred_diff + count * num_orientation_classes_,
                                          orientation_conf_bottom_diff + j * num_orientation_classes_);
                        ++count;
                    }
                    orientation_conf_bottom_diff += bottom[2]->offset(1);
                }
            } else {
                // The diff is already computed and stored.
                bottom[2]->ShareDiff(orientation_conf_pred_);
            }
        }
    }

  // After backward, remove match statistics.
  all_match_indices_.clear();
  all_neg_indices_.clear();
  all_orientation_match_indices_.clear();
  all_orientation_neg_indices_.clear();

}

INSTANTIATE_CLASS(MultiBoxLossLayer);
REGISTER_LAYER_CLASS(MultiBoxLoss);

}  // namespace caffe
