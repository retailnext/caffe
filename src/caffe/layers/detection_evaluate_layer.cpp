#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "caffe/layers/detection_evaluate_layer.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
void DetectionEvaluateLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const DetectionEvaluateParameter& detection_evaluate_param =
      this->layer_param_.detection_evaluate_param();
  CHECK(detection_evaluate_param.has_num_classes())
      << "Must provide num_classes.";
  CHECK(detection_evaluate_param.has_num_age_classes()) // Added by Dong Liu for MTL
      << "Must provide num_age_classes.";
  CHECK(detection_evaluate_param.has_num_gender_classes()) // Added by Dong Liu for MTL
        << "Must provide num_gender_classes.";
  num_classes_ = detection_evaluate_param.num_classes();
  num_age_classes_ = detection_evaluate_param.num_age_classes(); //Added by Dong liu for MTL
  num_gender_classes_ = detection_evaluate_param.num_gender_classes(); //Added by Dong Liu for MTL
  background_label_id_ = detection_evaluate_param.background_label_id();
  age_background_label_id_ = detection_evaluate_param.age_background_label_id(); //Added by Dong Liu for MTL
  gender_background_label_id_ = detection_evaluate_param.gender_background_label_id();

  overlap_threshold_ = detection_evaluate_param.overlap_threshold();
  CHECK_GT(overlap_threshold_, 0.) << "overlap_threshold must be non negative.";
  evaluate_difficult_gt_ = detection_evaluate_param.evaluate_difficult_gt();
  if (detection_evaluate_param.has_name_size_file()) {
    string name_size_file = detection_evaluate_param.name_size_file();
    std::ifstream infile(name_size_file.c_str());
    CHECK(infile.good())
        << "Failed to open name size file: " << name_size_file;
    // The file is in the following format:
    //    name height width
    //    ...
    string name;
    int height, width;
    while (infile >> name >> height >> width) {
      sizes_.push_back(std::make_pair(height, width));
    }
    infile.close();
  }
  count_ = 0;
  gender_count_ = 0;
  age_count_ = 0;
  // If there is no name_size_file provided, use normalized bbox to evaluate.
  use_normalized_bbox_ = sizes_.size() == 0;
}

template <typename Dtype>
void DetectionEvaluateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_LE(count_, sizes_.size());
  CHECK_EQ(bottom[0]->num(), 1);
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->width(), 11);
  CHECK_EQ(bottom[1]->num(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->width(), 10);
  /* Below commented by Dong Liu on March 10th 2017
  CHECK_EQ(bottom[2]->num(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->width(), 9); // change 8 into 10 by  Dong Liu for MTL. Ground Truth Label */

  // num() and channels() are 1.
  vector<int> top_shape(2, 1);
  int num_pos_classes = background_label_id_ == -1 ?
      num_classes_ : num_classes_ - 1;

  //int num_age_pos_classes = age_background_label_id_ == -1 ?
  //    num_age_classes_ : num_age_classes_ - 1;  // Added by Dong Liu for MTL

  //int num_gender_pos_classes = gender_background_label_id_ == -1 ?
  //    num_gender_classes_ : num_gender_classes_ - 1;  // Added by Dong Liu for MTL

  //top_shape.push_back(num_pos_classes + bottom[0]->height()); Commented by Dong Liu for MTL
  //top_shape.push_back(num_pos_classes +  num_gender_pos_classes + bottom[0]->height()); //Added by Dong Liu for MTL
  top_shape.push_back(num_pos_classes + bottom[0]->height());

  // Each row is a 5 dimension vector, which stores
  // [image_id, label, confidence, true_pos, false_pos, age, age_true_pos, age_false_pos, gender, gender_true_pos, gender_false_pos, age_score, gender_score]
  top_shape.push_back(13); // Change 5 into 13
  top[0]->Reshape(top_shape);


  //added on April 7th 2017
  // ********************************* Below added by Dong Liu for evaluting age
  // num() and channels() are 1.
  vector<int> age_top_shape(2, 1);

  //int num_pos_classes = background_label_id_ == -1 ?
  //      num_classes_ : num_classes_ - 1;

    //int num_age_pos_classes = age_background_label_id_ == -1 ?
    //    num_age_classes_ : num_age_classes_ - 1;  // Added by Dong Liu for MTL

  int num_age_pos_classes = age_background_label_id_ == -1 ?
         num_age_classes_ : num_age_classes_ - 1;  // Added by Dong Liu for MTL

    // top_shape.push_back(num_pos_classes + bottom[0]->height()); Commented by Dong Liu for MTL
  age_top_shape.push_back(num_age_pos_classes + bottom[0]->height()); //Added by Dong Liu for MTL
    // Each row is a 5 dimension vector, which stores
    // [image_id, label, confidence, true_pos, false_pos, age, age_true_pos, age_false_pos, gender, gender_true_pos, gender_false_pos, age_score, gender_score]
  age_top_shape.push_back(13); // Change 5 into 13
  top[1]->Reshape(age_top_shape);


  // ********************************* Below added by Dong Liu for evaluting gender
  // num() and channels() are 1.
  vector<int> gender_top_shape(2, 1);

  //int num_pos_classes = background_label_id_ == -1 ?
  //      num_classes_ : num_classes_ - 1;

    //int num_age_pos_classes = age_background_label_id_ == -1 ?
    //    num_age_classes_ : num_age_classes_ - 1;  // Added by Dong Liu for MTL

  int num_gender_pos_classes = gender_background_label_id_ == -1 ?
         num_gender_classes_ : num_gender_classes_ - 1;  // Added by Dong Liu for MTL

    // top_shape.push_back(num_pos_classes + bottom[0]->height()); Commented by Dong Liu for MTL
  gender_top_shape.push_back(num_gender_pos_classes + bottom[0]->height()); //Added by Dong Liu for MTL
    // Each row is a 5 dimension vector, which stores
    // [image_id, label, confidence, true_pos, false_pos, age, age_true_pos, age_false_pos, gender, gender_true_pos, gender_false_pos, age_score, gender_score]
  gender_top_shape.push_back(13); // Change 5 into 13
  top[2]->Reshape(gender_top_shape);

}

template <typename Dtype>
void DetectionEvaluateLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* det_data = bottom[0]->cpu_data();
  //const Dtype* gender_det_data = bottom[1]->cpu_data(); //commented by Dong Liu on March 10th 2017
  const Dtype* gt_data = bottom[1]->cpu_data(); // change 2 into 1 on March 10th 2017

  // Retrieve all detection results.
  map<int, LabelBBox> all_detections;
  GetDetectionResults(det_data, bottom[0]->height(), background_label_id_,
                      &all_detections);

  /* Below commented by Dong Liu on March 10th 2017
  //Retrieve all gender detection results.
  map<int, LabelBBox> all_gender_detections;
  GetGenderDetectionResults(gender_det_data, bottom[1]->height(), gender_background_label_id_,
		              &all_gender_detections); */

  //Added on April 7th 2017
  map<int, LabelBBox> all_age_detections;
  GetAgeDetectionResults(det_data, bottom[0]->height(), age_background_label_id_,
  		              &all_age_detections);

  //Below added by Dong Liu on March 10th 2017
  map<int, LabelBBox> all_gender_detections;
  GetGenderDetectionResults(det_data, bottom[0]->height(), gender_background_label_id_,
  		              &all_gender_detections);
  

  // Retrieve all ground truth (including difficult ones). change 2 into 1 on March 10th 2017
  map<int, LabelBBox> all_gt_bboxes;
  GetGroundTruth(gt_data, bottom[1]->height(), background_label_id_,
                 true, &all_gt_bboxes);

  // Retrieve all age ground truth (including difficult ones), Added by Dong Liu for MTL.
  //map<int, LabelBBox> all_age_gt_bboxes;
  //GetAgeGroundTruth(gt_data, bottom[1]->height(), age_background_label_id_,
    //             true, &all_age_gt_bboxes);


  //added on April 7th 2017
  map<int, LabelBBox> all_age_gt_bboxes;
  GetAgeGroundTruth(gt_data, bottom[1]->height(), age_background_label_id_,
                 true, &all_age_gt_bboxes);


  // Retrieve all gender ground truth (including difficult ones), Added by Dong Liu for MTL. Change 2 into 1 on March 10th 2017
  map<int, LabelBBox> all_gender_gt_bboxes;
  GetGenderGroundTruth(gt_data, bottom[1]->height(), gender_background_label_id_,
                 true, &all_gender_gt_bboxes);

  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0.), top_data);

  Dtype* age_top_data = top[1]->mutable_cpu_data();
  caffe_set(top[1]->count(), Dtype(0.), age_top_data);

  Dtype* gender_top_data = top[2]->mutable_cpu_data();
  caffe_set(top[2]->count(), Dtype(0.), gender_top_data);


  int num_det = 0;
  int age_num_det = 0;
  int gender_num_det = 0;


  // Insert number of ground truth for each label.
  map<int, int> num_pos;
  for (map<int, LabelBBox>::iterator it = all_gt_bboxes.begin();
       it != all_gt_bboxes.end(); ++it) {
    for (LabelBBox::iterator iit = it->second.begin(); iit != it->second.end();
         ++iit) {
      int count = 0;
      if (evaluate_difficult_gt_) {
        count = iit->second.size();
      } else {
        // Get number of non difficult ground truth.
        for (int i = 0; i < iit->second.size(); ++i) {
          if (!iit->second[i].difficult()) {
            ++count;
          }
        }
      }
      if (num_pos.find(iit->first) == num_pos.end()) {
        num_pos[iit->first] = count;
      } else {
        num_pos[iit->first] += count;
      }
    }
  }

  //Insert number of ground truth for each age label, Added by Dong Liu for MTL
  /*map<int, int> num_age_pos;
  for (map<int, LabelBBox>::iterator it = all_age_gt_bboxes.begin();
       it != all_age_gt_bboxes.end(); ++it) {
    for (LabelBBox::iterator iit = it->second.begin(); iit != it->second.end();
         ++iit) {
      int count = 0;
      if (evaluate_difficult_gt_) {
        count = iit->second.size();
      } else {
        // Get number of non difficult ground truth.
        for (int i = 0; i < iit->second.size(); ++i) {
          if (!iit->second[i].difficult()) {
            ++count;
          }
        }
      }
      if (num_age_pos.find(iit->first) == num_age_pos.end()) {
        num_age_pos[iit->first] = count;
      } else {
        num_age_pos[iit->first] += count;
      }
    }
  }*/

  //Added on April 7th 2017
  //Insert number of ground truth for each age label, Added by Dong Liu for MTL
  map<int, int> num_age_pos;
  for (map<int, LabelBBox>::iterator it = all_age_gt_bboxes.begin();
       it != all_age_gt_bboxes.end(); ++it) {
    for (LabelBBox::iterator iit = it->second.begin(); iit != it->second.end();
         ++iit) {
      int count = 0;
      if (evaluate_difficult_gt_) {
        count = iit->second.size();
      } else {
        // Get number of non difficult ground truth.
        for (int i = 0; i < iit->second.size(); ++i) {
          if (!iit->second[i].difficult()) {
            ++count;
          }
        }
      }
      if (num_age_pos.find(iit->first) == num_age_pos.end()) {
        num_age_pos[iit->first] = count;
      } else {
        num_age_pos[iit->first] += count;
      }
    }
  }

  //Insert number of ground truth for each gender label, Added by Dong Liu for MTL
  map<int, int> num_gender_pos;
  for (map<int, LabelBBox>::iterator it = all_gender_gt_bboxes.begin();
       it != all_gender_gt_bboxes.end(); ++it) {
    for (LabelBBox::iterator iit = it->second.begin(); iit != it->second.end();
         ++iit) {
      int count = 0;
      if (evaluate_difficult_gt_) {
        count = iit->second.size();
      } else {
        // Get number of non difficult ground truth.
        for (int i = 0; i < iit->second.size(); ++i) {
          if (!iit->second[i].difficult()) {
            ++count;
          }
        }
      }
      if (num_gender_pos.find(iit->first) == num_gender_pos.end()) {
        num_gender_pos[iit->first] = count;
      } else {
        num_gender_pos[iit->first] += count;
      }
    }
  }

  for (int c = 0; c < num_classes_; ++c) {
    if (c == background_label_id_) {
      continue;
    }
    top_data[num_det * 13] = -1;
    top_data[num_det * 13 + 1] = c;
    if (num_pos.find(c) == num_pos.end()) {
      top_data[num_det * 13 + 2] = 0;
    } else {
      top_data[num_det * 13 + 2] = num_pos.find(c)->second;
    }
    top_data[num_det * 13 + 3] = -1;
    top_data[num_det * 13 + 4] = -1;
    top_data[num_det * 13 + 5] = -1;
    top_data[num_det * 13 + 6] = -1;
    top_data[num_det * 13 + 7] = -1;
    top_data[num_det * 13 + 8] = -1;
    top_data[num_det * 13 + 9] = -1;
    top_data[num_det * 13 + 10] = -1;
    top_data[num_det * 13 + 11] = -1;
    top_data[num_det * 13 + 12] = -1;

    ++num_det;
  }

  //Added by Dong Liu for MTL -2 for age
  /*for (int c = 0; c < num_age_classes_; ++c) {
    if (c == age_background_label_id_) {
      continue;
    }
    top_data[num_det * 13] = -2;
    top_data[num_det * 13 + 5] = c;
    if (num_age_pos.find(c) == num_age_pos.end()) {
      top_data[num_det * 13 + 11] = 0;  //Added by Dong Liu for MTL
    } else {
      top_data[num_det * 13 + 11] = num_age_pos.find(c)->second; //Added by Dong Liu for MTL
    }
    top_data[num_det * 13 + 1] = -1;
    top_data[num_det * 13 + 2] = -1;
    top_data[num_det * 13 + 3] = -1;
    top_data[num_det * 13 + 4] = -1;
    top_data[num_det * 13 + 6] = -1;
    top_data[num_det * 13 + 7] = -1;
    top_data[num_det * 13 + 8] = -1;
    top_data[num_det * 13 + 9] = -1;
    top_data[num_det * 13 + 10] = -1;
    top_data[num_det * 13 + 12] = -1;

    ++num_det;
  }*/

  //Added on April 7th 2017

  //Added by Dong Liu for MTL -2 for age
  for (int c = 0; c < num_age_classes_; ++c) {
    if (c == age_background_label_id_) {
      continue;
    }
    age_top_data[age_num_det * 13] = -2;
    age_top_data[age_num_det * 13 + 5] = c;
    if (num_age_pos.find(c) == num_age_pos.end()) {
      age_top_data[age_num_det * 13 + 8] = 0;  //Added by Dong Liu for MTL
    } else {
      age_top_data[age_num_det * 13 + 8] = num_age_pos.find(c)->second; //Added by Dong Liu for MTL
    }
    age_top_data[age_num_det * 13 + 1] = -1;
    age_top_data[age_num_det * 13 + 2] = -1;
    age_top_data[age_num_det * 13 + 3] = -1;
    age_top_data[age_num_det * 13 + 4] = -1;
    age_top_data[age_num_det * 13 + 6] = -1;
    age_top_data[age_num_det * 13 + 7] = -1;
    age_top_data[age_num_det * 13 + 9] = -1;
    age_top_data[age_num_det * 13 + 10] = -1;
    age_top_data[age_num_det * 13 + 11] = -1;
    age_top_data[age_num_det * 13 + 12] = -1;

    ++age_num_det;
  }

  //Added by Dong Liu for MTL -3 for gender
  for (int c = 0; c < num_gender_classes_; ++c) {
    if (c == gender_background_label_id_) {
      continue;
    }
    gender_top_data[gender_num_det * 13] = -3;
    gender_top_data[gender_num_det * 13 + 9] = c;
    if (num_gender_pos.find(c) == num_gender_pos.end()) {
      gender_top_data[gender_num_det * 13 + 12] = 0;  //Added by Dong Liu for MTL
    } else {
      gender_top_data[gender_num_det * 13 + 12] = num_gender_pos.find(c)->second; //Added by Dong Liu for MTL
    }
    gender_top_data[gender_num_det * 13 + 1] = -1;
    gender_top_data[gender_num_det * 13 + 2] = -1;
    gender_top_data[gender_num_det * 13 + 3] = -1;
    gender_top_data[gender_num_det * 13 + 4] = -1;
    gender_top_data[gender_num_det * 13 + 5] = -1;
    gender_top_data[gender_num_det * 13 + 6] = -1;
    gender_top_data[gender_num_det * 13 + 7] = -1;
    gender_top_data[gender_num_det * 13 + 8] = -1;
    gender_top_data[gender_num_det * 13 + 10] = -1;
    gender_top_data[gender_num_det * 13 + 11] = -1;

    ++gender_num_det;
  }

  // Insert detection evaluate status.
  // [0: image_id, 1: label, 2: confidence, 3: true_pos, 4: false_pos, 5: gender, 6: gender_true_pos, 7: gender_false_pos, 8: gender_score]
  for (map<int, LabelBBox>::iterator it = all_detections.begin();
       it != all_detections.end(); ++it) {
    int image_id = it->first;
    LabelBBox& detections = it->second;
    if (all_gt_bboxes.find(image_id) == all_gt_bboxes.end()) {
      // No ground truth for current image. All detections become false_pos.
      for (LabelBBox::iterator iit = detections.begin();
           iit != detections.end(); ++iit) {
        int label = iit->first;
        const vector<NormalizedBBox>& bboxes = iit->second;
        for (int i = 0; i < bboxes.size(); ++i) {
          top_data[num_det * 13] = image_id;
          top_data[num_det * 13 + 1] = label;
          top_data[num_det * 13 + 2] = bboxes[i].score();
          top_data[num_det * 13 + 3] = 0;
          top_data[num_det * 13 + 4] = 1;
          //top_data[num_det * 13 + 5] = bboxes[i].age();
          //top_data[num_det * 13 + 6] = 0;
          //top_data[num_det * 13 + 7] = 1;
          top_data[num_det * 13 + 5] = 0; //bboxes[i].gender();
          top_data[num_det * 13 + 6] = 0;
          top_data[num_det * 13 + 7] = 1;
          //top_data[num_det * 13 + 11] = bboxes[i].ascore() ; //Added by Dong Liu for MTL
          top_data[num_det * 13  + 8] = 0; //bboxes[i].gscore(); //Added by Dong Liu for MTL
          top_data[num_det * 13 + 9] = 0; //bboxes[i].gender();
          top_data[num_det * 13 + 10] = 0;
          top_data[num_det * 13 + 11] = 1;
          //top_data[num_det * 13 + 11] = bboxes[i].ascore() ; //Added by Dong Liu for MTL
          top_data[num_det * 13  + 12] = 0; //bboxes[i].gscore(); //Added by Dong Liu for MTL

          ++num_det;
        }
      }
    } else {
      LabelBBox& label_bboxes = all_gt_bboxes.find(image_id)->second;
      for (LabelBBox::iterator iit = detections.begin();
           iit != detections.end(); ++iit) {
        int label = iit->first;
        vector<NormalizedBBox>& bboxes = iit->second;
        if (label_bboxes.find(label) == label_bboxes.end()) {
          // No ground truth for current label. All detections become false_pos.
          for (int i = 0; i < bboxes.size(); ++i) {
            top_data[num_det * 13] = image_id;
            top_data[num_det * 13 + 1] = label;
            top_data[num_det * 13 + 2] = bboxes[i].score();
            top_data[num_det * 13 + 3] = 0;
            top_data[num_det * 13 + 4] = 1;
            //top_data[num_det * 13 + 5] =  bboxes[i].age();
            //top_data[num_det * 13 + 6] = 0;
            //top_data[num_det * 13 + 7] = 1;
            top_data[num_det * 13 + 5] = 0; //bboxes[i].gender();
            top_data[num_det * 13 + 6] = 0;
            top_data[num_det * 13 + 7] = 1;
            //top_data[num_det * 13 + 11] = bboxes[i].ascore();
            top_data[num_det * 13 + 8] = 0 ; //bboxes[i].gscore();

            top_data[num_det * 13 + 9] = 0 ;
            top_data[num_det * 13 + 10] = 0 ;
            top_data[num_det * 13 + 11] = 1 ;
            top_data[num_det * 13 + 12] = 0 ;

            ++num_det;
          }
        } else {
          vector<NormalizedBBox>& gt_bboxes = label_bboxes.find(label)->second;
          // Scale ground truth if needed.
          if (!use_normalized_bbox_) {
            CHECK_LT(count_, sizes_.size());
            for (int i = 0; i < gt_bboxes.size(); ++i) {
              ScaleBBox(gt_bboxes[i], sizes_[count_].first,
                        sizes_[count_].second, &(gt_bboxes[i]));
            }
          }
          vector<bool> visited(gt_bboxes.size(), false);
          // Sort detections in descend order based on scores.
          std::sort(bboxes.begin(), bboxes.end(), SortBBoxDescend);
          for (int i = 0; i < bboxes.size(); ++i) {
            top_data[num_det * 13] = image_id;
            top_data[num_det * 13 + 1] = label;
            top_data[num_det * 13 + 2] = bboxes[i].score();
            //top_data[num_det * 13 + 5] = bboxes[i].age();
            top_data[num_det * 13 + 5] = 0; //bboxes[i].gender();
            //top_data[num_det * 13 + 11] = bboxes[i].ascore();
            top_data[num_det * 13 + 8] = 0; //bboxes[i].gscore();

            top_data[num_det * 13 + 9] = 0;
            top_data[num_det * 13 + 12] = 0;

            if (!use_normalized_bbox_) {
              ScaleBBox(bboxes[i], sizes_[count_].first, sizes_[count_].second,
                        &(bboxes[i]));
            }
            // Compare with each ground truth bbox.
            float overlap_max = -1;
            int jmax = -1;
            for (int j = 0; j < gt_bboxes.size(); ++j) {
              float overlap = JaccardOverlap(bboxes[i], gt_bboxes[j],
                                             use_normalized_bbox_);
              if (overlap > overlap_max) {
                overlap_max = overlap;
                jmax = j;
              }
            }
            if (overlap_max >= overlap_threshold_) {
              if (evaluate_difficult_gt_ ||
                  (!evaluate_difficult_gt_ && !gt_bboxes[jmax].difficult())) {
                if (!visited[jmax]) {
                  // true positive.
                  top_data[num_det * 13 + 3] = 1;
                  top_data[num_det * 13 + 4] = 0;
                  visited[jmax] = true;
                  // [0: image_id, 1: label, 2: confidence, 3: true_pos, 4: false_pos, 5: age, 6: age_true_pos, 7: age_false_pos, 8: gender, 9: gender_true_pos, 10: gender_false_pos, 11: age_score, 12: gender_score]

                  //top_data[num_det * 13 + 5] = bboxes[i].age();   // Added By dong Liu for MTL

                  /*if(abs(bboxes[i].age()-gt_bboxes[jmax].age())<10e-6) {
                	  top_data[num_det * 13 + 6] = 1;
                	  top_data[num_det * 13 + 7] = 0;
                  }
                  else {
                	 top_data[num_det * 13 + 6] = 0;
                	 top_data[num_det * 13 + 7] = 1;
                  }*/

                 //top_data[num_det * 13 + 8] = bboxes[i].gender();
                 /*if(abs(bboxes[i].gender()-gt_bboxes[jmax].gender())<10e-6) {
                	 top_data[num_det * 9 + 6] = 1;
                	 top_data[num_det * 9 + 7] = 0;
                 }
                 else {
                	 top_data[num_det * 9 + 6] = 0;
                	 top_data[num_det * 9 + 7] = 1;
                 }*/
                  top_data[num_det * 13 + 6] = 0;
                  top_data[num_det * 13 + 7] = 0;

                  top_data[num_det * 13 + 10] = 0;
                  top_data[num_det * 13 + 11] = 0;
                } else {
                  // false positive (multiple detection).
                  top_data[num_det * 13 + 3] = 0;
                  top_data[num_det * 13 + 4] = 1;
                  //top_data[num_det * 13 + 5] = 0;
                  top_data[num_det * 13 + 6] = 0;
                  top_data[num_det * 13 + 7] = 1;
                  //top_data[num_det * 13 + 8] = 0;
                  //top_data[num_det * 13 + 9] = 0;
                  //top_data[num_det * 13 + 10] = 1;
                  top_data[num_det * 13 + 10] = 0;
                  top_data[num_det * 13 + 11] = 1;
                }
              }
            } else {
              // false positive.
              top_data[num_det * 13 + 3] = 0;
              top_data[num_det * 13 + 4] = 1;
              //top_data[num_det * 13 + 5] = 0;
              top_data[num_det * 13 + 6] = 0;
              top_data[num_det * 13 + 7] = 1;

              top_data[num_det * 13 + 10] = 0;
              top_data[num_det * 13 + 11] = 1;

              //top_data[num_det * 13 + 8] = 0;
              //top_data[num_det * 13 + 9] = 0;
              //top_data[num_det * 13 + 10] = 1;
            }
            ++num_det;
          }
        }
      }
    }
    if (sizes_.size() > 0) {
      ++count_;
      if (count_ == sizes_.size()) {
        // reset count after a full iterations through the DB.
        count_ = 0;
      }
    }
  }// for each detection result

  //Added on April 7th 2017 for age MTL
   for (map<int, LabelBBox>::iterator it = all_age_detections.begin();
          it != all_age_detections.end(); ++it) {
       int image_id = it->first;
       LabelBBox& detections = it->second;
       if (all_age_gt_bboxes.find(image_id) == all_age_gt_bboxes.end()) {
         // No ground truth for current image. All detections become false_pos.
         for (LabelBBox::iterator iit = detections.begin();
              iit != detections.end(); ++iit) {
           int label = iit->first;
           const vector<NormalizedBBox>& bboxes = iit->second;
           for (int i = 0; i < bboxes.size(); ++i) {
             age_top_data[age_num_det * 13] = image_id;
             age_top_data[age_num_det * 13 + 1] = 0; //label;
             age_top_data[age_num_det * 13 + 2] = 0 ; //bboxes[i].score();
             age_top_data[age_num_det * 13 + 3] = 0;
             age_top_data[age_num_det * 13 + 4] = 0;
             //top_data[num_det * 13 + 5] = bboxes[i].age();
             //top_data[num_det * 13 + 6] = 0;
             //top_data[num_det * 13 + 7] = 1;
             age_top_data[age_num_det * 13 + 5] = label; //bboxes[i].gender();
             age_top_data[age_num_det * 13 + 6] = 0;
             age_top_data[age_num_det * 13 + 7] = 1;
             //top_data[num_det * 13 + 11] = bboxes[i].ascore() ; //Added by Dong Liu for MTL
             age_top_data[age_num_det * 13  + 8] = bboxes[i].ascore(); //bboxes[i].gscore(); //Added by Dong Liu for MTL

             age_top_data[age_num_det * 13 + 9] = 0;
             age_top_data[age_num_det * 13 + 10] = 0 ;
             age_top_data[age_num_det * 13 + 11] = 0;
             age_top_data[age_num_det * 13 + 12] = 0;

             ++age_num_det;
           }
         }
       } else {
         LabelBBox& label_bboxes = all_age_gt_bboxes.find(image_id)->second;
         for (LabelBBox::iterator iit = detections.begin();
              iit != detections.end(); ++iit) {
           int label = iit->first;
           vector<NormalizedBBox>& bboxes = iit->second;
           if (label_bboxes.find(label) == label_bboxes.end()) {
             // No ground truth for current label. All detections become false_pos.
             for (int i = 0; i < bboxes.size(); ++i) {
               age_top_data[age_num_det * 13] = image_id;
               age_top_data[age_num_det * 13 + 1] = 0; //label;
               age_top_data[age_num_det * 13 + 2] = 0; //bboxes[i].score();
               age_top_data[age_num_det * 13 + 3] = 0;
               age_top_data[age_num_det * 13 + 4] = 0;
               //top_data[num_det * 13 + 5] =  bboxes[i].age();
               //top_data[num_det * 13 + 6] = 0;
               //top_data[num_det * 13 + 7] = 1;
               age_top_data[age_num_det * 13 + 5] = label; //bboxes[i].gender();
               age_top_data[age_num_det * 13 + 6] = 0;
               age_top_data[age_num_det * 13 + 7] = 1;
               //top_data[num_det * 13 + 11] = bboxes[i].ascore();
               age_top_data[age_num_det * 13 + 8] = bboxes[i].ascore(); //bboxes[i].gscore();

               age_top_data[age_num_det * 13 + 9] = 0;
               age_top_data[age_num_det * 13 + 10] = 0 ;
               age_top_data[age_num_det * 13 + 11] = 0;
               age_top_data[age_num_det * 13 + 12] = 0;

               ++age_num_det;
             }
           } else {
             vector<NormalizedBBox>& gt_bboxes = label_bboxes.find(label)->second;
             // Scale ground truth if needed.
             if (!use_normalized_bbox_) {
               CHECK_LT(age_count_, sizes_.size());
               for (int i = 0; i < gt_bboxes.size(); ++i) {
                 ScaleBBox(gt_bboxes[i], sizes_[age_count_].first,
                           sizes_[age_count_].second, &(gt_bboxes[i]));
               }
             }
             vector<bool> visited(gt_bboxes.size(), false);
             // Sort detections in descend order based on scores.
             std::sort(bboxes.begin(), bboxes.end(), SortBBoxDescend);
             for (int i = 0; i < bboxes.size(); ++i) {
               age_top_data[age_num_det * 13] = image_id;
               age_top_data[age_num_det * 13 + 1] = 0; //label;
               age_top_data[age_num_det * 13 + 2] = 0; //bboxes[i].score();
               //top_data[num_det * 13 + 5] = bboxes[i].age();
               age_top_data[age_num_det * 13 + 5] = label; //bboxes[i].gender();
               //top_data[num_det * 13 + 11] = bboxes[i].ascore();
               age_top_data[age_num_det * 13 + 8] = bboxes[i].ascore();
               if (!use_normalized_bbox_) {
                 ScaleBBox(bboxes[i], sizes_[age_count_].first, sizes_[age_count_].second,
                           &(bboxes[i]));
               }
               // Compare with each ground truth bbox.
               float overlap_max = -1;
               int jmax = -1;
               for (int j = 0; j < gt_bboxes.size(); ++j) {
                 float overlap = JaccardOverlap(bboxes[i], gt_bboxes[j],
                                                use_normalized_bbox_);
                 if (overlap > overlap_max) {
                   overlap_max = overlap;
                   jmax = j;
                 }
               }
               if (overlap_max >= overlap_threshold_) {
                 if (evaluate_difficult_gt_ ||
                     (!evaluate_difficult_gt_ && !gt_bboxes[jmax].difficult())) {
                   if (!visited[jmax]) {
                     // true positive.
                     age_top_data[age_num_det * 13 + 3] = 0;
                     age_top_data[age_num_det * 13 + 4] = 0;
                     age_top_data[age_num_det * 13 + 10] = 0;
                     age_top_data[age_num_det * 13 + 11] = 0;

                     visited[jmax] = true;
                     // [0: image_id, 1: label, 2: confidence, 3: true_pos, 4: false_pos, 5: age, 6: age_true_pos, 7: age_false_pos, 8: gender, 9: gender_true_pos, 10: gender_false_pos, 11: age_score, 12: gender_score]

                     //top_data[num_det * 13 + 5] = bboxes[i].age();   // Added By dong Liu for MTL

                     /*if(abs(bboxes[i].age()-gt_bboxes[jmax].age())<10e-6) {
                   	  top_data[num_det * 13 + 6] = 1;
                   	  top_data[num_det * 13 + 7] = 0;
                     }
                     else {
                   	 top_data[num_det * 13 + 6] = 0;
                   	 top_data[num_det * 13 + 7] = 1;
                     }*/

                    //top_data[num_det * 13 + 8] = bboxes[i].gender();
                    if(abs(bboxes[i].age()-gt_bboxes[jmax].age())<10e-6) {
                   	 /*top_data[num_det * 9 + 6] = 1;
                   	 top_data[num_det * 9 + 7] = 0;
                    }
                    else {
                   	 top_data[num_det * 9 + 6] = 0;
                   	 top_data[num_det * 9 + 7] = 1;
                    }*/
                     age_top_data[age_num_det * 13 + 6] = 1;
                     age_top_data[age_num_det * 13 + 7] = 0;

                   } else {
                     // false positive (multiple detection).
                     age_top_data[age_num_det * 13 + 3] = 0;
                     age_top_data[age_num_det * 13 + 4] = 1;
                     //top_data[num_det * 13 + 5] = 0;
                     age_top_data[age_num_det * 13 + 6] = 0;
                     age_top_data[age_num_det * 13 + 7] = 1;

                     age_top_data[age_num_det * 13 + 10] = 0;
                     age_top_data[age_num_det * 13 + 11] = 1;

                     //top_data[num_det * 13 + 8] = 0;
                     //top_data[num_det * 13 + 9] = 0;
                     //top_data[num_det * 13 + 10] = 1;
                   }
                 }
               } else {
                 // false positive.
                 age_top_data[age_num_det * 13 + 3] = 0;
                 age_top_data[age_num_det * 13 + 4] = 1;
                 //top_data[num_det * 13 + 5] = 0;
                 age_top_data[age_num_det * 13 + 6] = 0;
                 age_top_data[age_num_det * 13 + 7] = 1;


                 age_top_data[age_num_det * 13 + 10] = 0;
                 age_top_data[age_num_det * 13 + 11] = 1;

                 //top_data[num_det * 13 + 8] = 0;
                 //top_data[num_det * 13 + 9] = 0;
                 //top_data[num_det * 13 + 10] = 1;
               }
               ++age_num_det;
             }
           }
         }
       }
       if (sizes_.size() > 0) {
         ++age_count_;
         if (age_count_ == sizes_.size()) {
           // reset count after a full iterations through the DB.
           age_count_ = 0;
         }
       }
     }// for each detection result

  }

  // Bolow added by Dong Liu on March 10th 2017 
  // Insert gender detection evaluate status.
     // [0: image_id, 1: label, 2: confidence, 3: true_pos, 4: false_pos, 5: gender, 6: gender_true_pos, 7: gender_false_pos, 8: gender_score]
     for (map<int, LabelBBox>::iterator it = all_gender_detections.begin();
          it != all_gender_detections.end(); ++it) {
       int image_id = it->first;
       LabelBBox& detections = it->second;
       if (all_gender_gt_bboxes.find(image_id) == all_gender_gt_bboxes.end()) {
         // No ground truth for current image. All detections become false_pos.
         for (LabelBBox::iterator iit = detections.begin();
              iit != detections.end(); ++iit) {
           int label = iit->first;
           const vector<NormalizedBBox>& bboxes = iit->second;
           for (int i = 0; i < bboxes.size(); ++i) {
             gender_top_data[gender_num_det * 13] = image_id;
             gender_top_data[gender_num_det * 13 + 1] = 0; //label;
             gender_top_data[gender_num_det * 13 + 2] = 0 ; //bboxes[i].score();
             gender_top_data[gender_num_det * 13 + 3] = 0;
             gender_top_data[gender_num_det * 13 + 4] = 0;
             gender_top_data[gender_num_det * 13 + 5] = 0; //bboxes[i].gender();
             gender_top_data[gender_num_det * 13 + 6] = 0;
             gender_top_data[gender_num_det * 13 + 7] = 0;
             gender_top_data[gender_num_det * 13 + 8] = 0; //bboxes[i].gscore(); //Added by Dong Liu for MTL

             gender_top_data[gender_num_det * 13 + 9] = label; //bboxes[i].gender();
             gender_top_data[gender_num_det * 13 + 10] = 0;
             gender_top_data[gender_num_det * 13 + 11] = 1;
             gender_top_data[gender_num_det * 13  + 12] = bboxes[i].gscore(); //bboxes[i].gscore(); //Added by Dong Liu for MTL

             ++gender_num_det;
           }
         }
       } else {
         LabelBBox& label_bboxes = all_gender_gt_bboxes.find(image_id)->second;
         for (LabelBBox::iterator iit = detections.begin();
              iit != detections.end(); ++iit) {
           int label = iit->first;
           vector<NormalizedBBox>& bboxes = iit->second;
           if (label_bboxes.find(label) == label_bboxes.end()) {
             // No ground truth for current label. All detections become false_pos.
             for (int i = 0; i < bboxes.size(); ++i) {
               gender_top_data[gender_num_det * 13] = image_id;
               gender_top_data[gender_num_det * 13 + 1] = 0; //label;
               gender_top_data[gender_num_det * 13 + 2] = 0; //bboxes[i].score();
               gender_top_data[gender_num_det * 13 + 3] = 0;
               gender_top_data[gender_num_det * 13 + 4] = 0;
               gender_top_data[gender_num_det * 13 + 5] = 0; //bboxes[i].gender();
               gender_top_data[gender_num_det * 13 + 6] = 0;
               gender_top_data[gender_num_det * 13 + 7] = 0;
               gender_top_data[gender_num_det * 13 + 8] = 0; //bboxes[i].gscore(); //Added by Dong Liu for MTL

               gender_top_data[gender_num_det * 13 + 9] = label; //bboxes[i].gender();
               gender_top_data[gender_num_det * 13 + 10] = 0;
               gender_top_data[gender_num_det * 13 + 11] = 1;
               //top_data[num_det * 13 + 11] = bboxes[i].ascore();
               gender_top_data[gender_num_det * 13 + 12] = bboxes[i].gscore(); //bboxes[i].gscore();

               ++gender_num_det;
             }
           } else {
             vector<NormalizedBBox>& gt_bboxes = label_bboxes.find(label)->second;
             // Scale ground truth if needed.
             if (!use_normalized_bbox_) {
               CHECK_LT(gender_count_, sizes_.size());
               for (int i = 0; i < gt_bboxes.size(); ++i) {
                 ScaleBBox(gt_bboxes[i], sizes_[gender_count_].first,
                           sizes_[gender_count_].second, &(gt_bboxes[i]));
               }
             }
             vector<bool> visited(gt_bboxes.size(), false);
             // Sort detections in descend order based on scores.
             std::sort(bboxes.begin(), bboxes.end(), SortBBoxDescend);
             for (int i = 0; i < bboxes.size(); ++i) {
               gender_top_data[gender_num_det * 13] = image_id;
               gender_top_data[gender_num_det * 13 + 1] = 0; //label;
               gender_top_data[gender_num_det * 13 + 2] = 0; //bboxes[i].score();
               gender_top_data[gender_num_det * 13 + 9] = label; //bboxes[i].gender();
               //top_data[num_det * 13 + 11] = bboxes[i].ascore();
               gender_top_data[gender_num_det * 13 + 12] = bboxes[i].gscore();
               if (!use_normalized_bbox_) {
                 ScaleBBox(bboxes[i], sizes_[gender_count_].first, sizes_[gender_count_].second,
                           &(bboxes[i]));
               }
               // Compare with each ground truth bbox.
               float overlap_max = -1;
               int jmax = -1;
               for (int j = 0; j < gt_bboxes.size(); ++j) {
                 float overlap = JaccardOverlap(bboxes[i], gt_bboxes[j],
                                                use_normalized_bbox_);
                 if (overlap > overlap_max) {
                   overlap_max = overlap;
                   jmax = j;
                 }
               }
               if (overlap_max >= overlap_threshold_) {
                 if (evaluate_difficult_gt_ ||
                     (!evaluate_difficult_gt_ && !gt_bboxes[jmax].difficult())) {
                   if (!visited[jmax]) {
                     // true positive.
                     gender_top_data[gender_num_det * 13 + 3] = 0;
                     gender_top_data[gender_num_det * 13 + 4] = 0;
                     gender_top_data[gender_num_det * 13 + 6] = 0;
                     gender_top_data[gender_num_det * 13 + 7] = 0;

                     visited[jmax] = true;
                     // [0: image_id, 1: label, 2: confidence, 3: true_pos, 4: false_pos, 5: age, 6: age_true_pos, 7: age_false_pos, 8: gender, 9: gender_true_pos, 10: gender_false_pos, 11: age_score, 12: gender_score]

                     //top_data[num_det * 13 + 5] = bboxes[i].age();   // Added By dong Liu for MTL

                     /*if(abs(bboxes[i].age()-gt_bboxes[jmax].age())<10e-6) {
                   	  top_data[num_det * 13 + 6] = 1;
                   	  top_data[num_det * 13 + 7] = 0;
                     }
                     else {
                   	 top_data[num_det * 13 + 6] = 0;
                   	 top_data[num_det * 13 + 7] = 1;
                     }*/

                    //top_data[num_det * 13 + 8] = bboxes[i].gender();
                    if(abs(bboxes[i].gender()-gt_bboxes[jmax].gender())<10e-6) {
                   	 /*top_data[num_det * 9 + 6] = 1;
                   	 top_data[num_det * 9 + 7] = 0;
                    }
                    else {
                   	 top_data[num_det * 9 + 6] = 0;
                   	 top_data[num_det * 9 + 7] = 1;
                    }*/
                     gender_top_data[gender_num_det * 13 + 10] = 1;
                     gender_top_data[gender_num_det * 13 + 11] = 0;

                   } else {
                     // false positive (multiple detection).
                     gender_top_data[gender_num_det * 13 + 3] = 0;
                     gender_top_data[gender_num_det * 13 + 4] = 1;
                     //top_data[num_det * 13 + 5] = 0;
                     gender_top_data[gender_num_det * 13 + 6] = 0;
                     gender_top_data[gender_num_det * 13 + 7] = 1;

                     gender_top_data[gender_num_det * 13 + 10] = 0;
                     gender_top_data[gender_num_det * 13 + 11] = 1;
                   }
                 }
               } else {
                 // false positive.
                 gender_top_data[gender_num_det * 13 + 3] = 0;
                 gender_top_data[gender_num_det * 13 + 4] = 1;
                 //top_data[num_det * 13 + 5] = 0;
                 gender_top_data[gender_num_det * 13 + 6] = 0;
                 gender_top_data[gender_num_det * 13 + 7] = 1;

                 gender_top_data[gender_num_det * 13 + 10] = 0;
                 gender_top_data[gender_num_det * 13 + 11] = 1;

               }
               ++gender_num_det;
             }
           }
         }
       }
       if (sizes_.size() > 0) {
         ++gender_count_;
         if (gender_count_ == sizes_.size()) {
           // reset count after a full iterations through the DB.
           gender_count_ = 0;
         }
       }
     }// for each detection result
  
  }
  
/*Below commented by Dong Liu on March 10th 2017
  
  // Insert gender detection evaluate status.
    // [0: image_id, 1: label, 2: confidence, 3: true_pos, 4: false_pos, 5: gender, 6: gender_true_pos, 7: gender_false_pos, 8: gender_score]
    for (map<int, LabelBBox>::iterator it = all_gender_detections.begin();
         it != all_gender_detections.end(); ++it) {
      int image_id = it->first;
      LabelBBox& detections = it->second;
      if (all_gender_gt_bboxes.find(image_id) == all_gender_gt_bboxes.end()) {
        // No ground truth for current image. All detections become false_pos.
        for (LabelBBox::iterator iit = detections.begin();
             iit != detections.end(); ++iit) {
          int label = iit->first;
          const vector<NormalizedBBox>& bboxes = iit->second;
          for (int i = 0; i < bboxes.size(); ++i) {
            gender_top_data[gender_num_det * 9] = image_id;
            gender_top_data[gender_num_det * 9 + 1] = 0; //label;
            gender_top_data[gender_num_det * 9 + 2] = 0 ; //bboxes[i].score();
            gender_top_data[gender_num_det * 9 + 3] = 0;
            gender_top_data[gender_num_det * 9 + 4] = 0;
            //top_data[num_det * 13 + 5] = bboxes[i].age();
            //top_data[num_det * 13 + 6] = 0;
            //top_data[num_det * 13 + 7] = 1;
            gender_top_data[gender_num_det * 9 + 5] = label; //bboxes[i].gender();
            gender_top_data[gender_num_det * 9 + 6] = 0;
            gender_top_data[gender_num_det * 9 + 7] = 1;
            //top_data[num_det * 13 + 11] = bboxes[i].ascore() ; //Added by Dong Liu for MTL
            gender_top_data[gender_num_det * 9  + 8] = bboxes[i].gscore(); //bboxes[i].gscore(); //Added by Dong Liu for MTL

            ++gender_num_det;
          }
        }
      } else {
        LabelBBox& label_bboxes = all_gender_gt_bboxes.find(image_id)->second;
        for (LabelBBox::iterator iit = detections.begin();
             iit != detections.end(); ++iit) {
          int label = iit->first;
          vector<NormalizedBBox>& bboxes = iit->second;
          if (label_bboxes.find(label) == label_bboxes.end()) {
            // No ground truth for current label. All detections become false_pos.
            for (int i = 0; i < bboxes.size(); ++i) {
              gender_top_data[gender_num_det * 9] = image_id;
              gender_top_data[gender_num_det * 9 + 1] = 0; //label;
              gender_top_data[gender_num_det * 9 + 2] = 0; //bboxes[i].score();
              gender_top_data[gender_num_det * 9 + 3] = 0;
              gender_top_data[gender_num_det * 9 + 4] = 0;
              //top_data[num_det * 13 + 5] =  bboxes[i].age();
              //top_data[num_det * 13 + 6] = 0;
              //top_data[num_det * 13 + 7] = 1;
              gender_top_data[gender_num_det * 9 + 5] = label; //bboxes[i].gender();
              gender_top_data[gender_num_det * 9 + 6] = 0;
              gender_top_data[gender_num_det * 9 + 7] = 1;
              //top_data[num_det * 13 + 11] = bboxes[i].ascore();
              gender_top_data[gender_num_det * 9 + 8] = bboxes[i].gscore(); //bboxes[i].gscore();

              ++gender_num_det;
            }
          } else {
            vector<NormalizedBBox>& gt_bboxes = label_bboxes.find(label)->second;
            // Scale ground truth if needed.
            if (!use_normalized_bbox_) {
              CHECK_LT(gender_count_, sizes_.size());
              for (int i = 0; i < gt_bboxes.size(); ++i) {
                ScaleBBox(gt_bboxes[i], sizes_[gender_count_].first,
                          sizes_[gender_count_].second, &(gt_bboxes[i]));
              }
            }
            vector<bool> visited(gt_bboxes.size(), false);
            // Sort detections in descend order based on scores.
            std::sort(bboxes.begin(), bboxes.end(), SortBBoxDescend);
            for (int i = 0; i < bboxes.size(); ++i) {
              gender_top_data[gender_num_det * 9] = image_id;
              gender_top_data[gender_num_det * 9 + 1] = 0; //label;
              gender_top_data[gender_num_det * 9 + 2] = 0; //bboxes[i].score();
              //top_data[num_det * 13 + 5] = bboxes[i].age();
              gender_top_data[gender_num_det * 9 + 5] = label; //bboxes[i].gender();
              //top_data[num_det * 13 + 11] = bboxes[i].ascore();
              gender_top_data[gender_num_det * 9 + 8] = bboxes[i].gscore();
              if (!use_normalized_bbox_) {
                ScaleBBox(bboxes[i], sizes_[gender_count_].first, sizes_[gender_count_].second,
                          &(bboxes[i]));
              }
              // Compare with each ground truth bbox.
              float overlap_max = -1;
              int jmax = -1;
              for (int j = 0; j < gt_bboxes.size(); ++j) {
                float overlap = JaccardOverlap(bboxes[i], gt_bboxes[j],
                                               use_normalized_bbox_);
                if (overlap > overlap_max) {
                  overlap_max = overlap;
                  jmax = j;
                }
              }
              if (overlap_max >= overlap_threshold_) {
                if (evaluate_difficult_gt_ ||
                    (!evaluate_difficult_gt_ && !gt_bboxes[jmax].difficult())) {
                  if (!visited[jmax]) {
                    // true positive.
                    gender_top_data[gender_num_det * 9 + 3] = 0;
                    gender_top_data[gender_num_det * 9 + 4] = 0;
                    visited[jmax] = true;
                    // [0: image_id, 1: label, 2: confidence, 3: true_pos, 4: false_pos, 5: age, 6: age_true_pos, 7: age_false_pos, 8: gender, 9: gender_true_pos, 10: gender_false_pos, 11: age_score, 12: gender_score]

                    //top_data[num_det * 13 + 5] = bboxes[i].age();   // Added By dong Liu for MTL

                    if(abs(bboxes[i].age()-gt_bboxes[jmax].age())<10e-6) {
                  	  top_data[num_det * 13 + 6] = 1;
                  	  top_data[num_det * 13 + 7] = 0;
                    }
                    else {
                  	 top_data[num_det * 13 + 6] = 0;
                  	 top_data[num_det * 13 + 7] = 1;
                    }

                   //top_data[num_det * 13 + 8] = bboxes[i].gender();
                   if(abs(bboxes[i].gender()-gt_bboxes[jmax].gender())<10e-6) {
                  	 top_data[num_det * 9 + 6] = 1;
                  	 top_data[num_det * 9 + 7] = 0;
                   }
                   else {
                  	 top_data[num_det * 9 + 6] = 0;
                  	 top_data[num_det * 9 + 7] = 1;
                   }
                    gender_top_data[gender_num_det * 9 + 6] = 1;
                    gender_top_data[gender_num_det * 9 + 7] = 0;

                  } else {
                    // false positive (multiple detection).
                    gender_top_data[gender_num_det * 9 + 3] = 0;
                    gender_top_data[gender_num_det * 9 + 4] = 1;
                    //top_data[num_det * 13 + 5] = 0;
                    gender_top_data[gender_num_det * 9 + 6] = 0;
                    gender_top_data[gender_num_det * 9 + 7] = 1;
                    //top_data[num_det * 13 + 8] = 0;
                    //top_data[num_det * 13 + 9] = 0;
                    //top_data[num_det * 13 + 10] = 1;
                  }
                }
              } else {
                // false positive.
                gender_top_data[gender_num_det * 9 + 3] = 0;
                gender_top_data[gender_num_det * 9 + 4] = 1;
                //top_data[num_det * 13 + 5] = 0;
                gender_top_data[gender_num_det * 9 + 6] = 0;
                gender_top_data[gender_num_det * 9 + 7] = 1;
                //top_data[num_det * 13 + 8] = 0;
                //top_data[num_det * 13 + 9] = 0;
                //top_data[num_det * 13 + 10] = 1;
              }
              ++gender_num_det;
            }
          }
        }
      }
      if (sizes_.size() > 0) {
        ++gender_count_;
        if (gender_count_ == sizes_.size()) {
          // reset count after a full iterations through the DB.
          gender_count_ = 0;
        }
      }
    }// for each detection result
   
 Above commented by Dong Liu on March 10th 2017    */


} // end of function "forward_cpu"

INSTANTIATE_CLASS(DetectionEvaluateLayer);
REGISTER_LAYER_CLASS(DetectionEvaluate);

}  // namespace caffe
