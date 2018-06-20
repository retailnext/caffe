#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/detection_output_layer.hpp"

namespace caffe {

template <typename Dtype>
void DetectionOutputLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->gpu_data();
  const Dtype* prior_data = bottom[4]->gpu_data(); //change 2 into 4 by Dong Liu for MTL

  const int num = bottom[0]->num();

  // Decode predictions.
  Blob<Dtype> bbox_preds;
  bbox_preds.ReshapeLike(*(bottom[0]));
  Dtype* bbox_data = bbox_preds.mutable_gpu_data();
  const int loc_count = bbox_preds.count();
  DecodeBBoxesGPU<Dtype>(loc_count, loc_data, prior_data, code_type_,
      variance_encoded_in_target_, num_priors_, share_location_,
      num_loc_classes_, background_label_id_, bbox_data);
  // Retrieve all decoded location predictions.
  const Dtype* bbox_cpu_data = bbox_preds.cpu_data();
  vector<LabelBBox> all_decode_bboxes;
  GetLocPredictions(bbox_cpu_data, num, num_priors_, num_loc_classes_,
      share_location_, &all_decode_bboxes);

  // Retrieve all confidences.
  const Dtype* conf_data;
  Blob<Dtype> conf_permute;
  conf_permute.ReshapeLike(*(bottom[1]));
  Dtype* conf_permute_data = conf_permute.mutable_gpu_data();
  PermuteDataGPU<Dtype>(conf_permute.count(), bottom[1]->gpu_data(),
      num_classes_, num_priors_, 1, conf_permute_data);
  conf_data = conf_permute.cpu_data();
  const bool class_major = true;
  vector<map<int, vector<float> > > all_conf_scores;
  GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
      class_major, &all_conf_scores);


  // Retrieval all age confidences added by Dong Liu for MTL
  /*const Dtype* age_data;
  Blob<Dtype> age_permute;
  age_permute.ReshapeLike(*(bottom[2]));
  Dtype* age_permute_data = age_permute.mutable_gpu_data();
  PermuteDataGPU<Dtype>(age_permute.count(), bottom[2]->gpu_data(),
      num_age_classes_, num_priors_, 1, age_permute_data);
  age_data = age_permute.cpu_data();
  vector<map<int, vector<float> > > all_age_scores;
  GetAGScores(age_data, num, num_priors_, num_age_classes_,
                        &all_age_scores);*/

  // Retrieval all age confidences added by Dong Liu for MTL
  const Dtype* age_data;
  Blob<Dtype> age_permute;
  age_permute.ReshapeLike(*(bottom[2]));
  Dtype* age_permute_data = age_permute.mutable_gpu_data();
  PermuteDataGPU<Dtype>(age_permute.count(), bottom[2]->gpu_data(),
      num_age_classes_, num_priors_, 1, age_permute_data);
  age_data = age_permute.cpu_data();
  vector<map<int, vector<float> > > all_age_scores;
  GetAGScores(age_data, num, num_priors_, num_age_classes_,
                           &all_age_scores); // added on April 6th 2017

  // Retrieval all gender confidences added by Dong Liu for MTL
  const Dtype* gender_data;
  Blob<Dtype> gender_permute;
  gender_permute.ReshapeLike(*(bottom[3]));
  Dtype* gender_permute_data = gender_permute.mutable_gpu_data();
  PermuteDataGPU<Dtype>(gender_permute.count(), bottom[3]->gpu_data(),
      num_gender_classes_, num_priors_, 1, gender_permute_data);
  gender_data = gender_permute.cpu_data();
  vector<map<int, vector<float> > > all_gender_scores;
  GetAGScores(gender_data, num, num_priors_, num_gender_classes_,
                           &all_gender_scores); // uncommented on MArch 10th 2017
  //GetConfidenceScores(gender_data, num, num_priors_, num_gender_classes_, //commented on March 10th 2017
  //		  class_major, &all_gender_scores);


  int num_kept = 0;
  vector<map<int, vector<int> > > all_indices;
  for (int i = 0; i < num; ++i) {
    const LabelBBox& decode_bboxes = all_decode_bboxes[i];
    const map<int, vector<float> >& conf_scores = all_conf_scores[i];
    map<int, vector<int> > indices;
    int num_det = 0;
    for (int c = 0; c < num_classes_; ++c) {
      if (c == background_label_id_) {
        // Ignore background class.
        continue;
      }
      if (conf_scores.find(c) == conf_scores.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find confidence predictions for label " << c;
      }
      const vector<float>& scores = conf_scores.find(c)->second;
      int label = share_location_ ? -1 : c;
      if (decode_bboxes.find(label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for label " << label;
        continue;
      }
      const vector<NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;
      ApplyNMSFast(bboxes, scores, confidence_threshold_, nms_threshold_,
          top_k_, &(indices[c]));
      num_det += indices[c].size();
    } // collect indices after nms
    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
      vector<pair<float, pair<int, int> > > score_index_pairs;
      for (map<int, vector<int> >::iterator it = indices.begin();
           it != indices.end(); ++it) {
        int label = it->first;
        const vector<int>& label_indices = it->second;
        if (conf_scores.find(label) == conf_scores.end()) {
          // Something bad happened for current label.
          LOG(FATAL) << "Could not find location predictions for " << label;
          continue;
        }
        const vector<float>& scores = conf_scores.find(label)->second;
        for (int j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          CHECK_LT(idx, scores.size());
          score_index_pairs.push_back(std::make_pair(
                  scores[idx], std::make_pair(label, idx)));
        }
      }
      // Keep top k results per image.
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScorePairDescend<pair<int, int> >);
      score_index_pairs.resize(keep_top_k_);
      // Store the new indices.
      map<int, vector<int> > new_indices;
      for (int j = 0; j < score_index_pairs.size(); ++j) {
        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        new_indices[label].push_back(idx);
      }
      all_indices.push_back(new_indices);
      num_kept += keep_top_k_;
    } else {
      all_indices.push_back(indices);
      num_kept += num_det;
    }
  } // for each image

  vector<int> top_shape(2, 1);
  top_shape.push_back(num_kept);
  top_shape.push_back(11); //Change 9 into 11 on April 7th 2017
  if (num_kept == 0) {
    LOG(INFO) << "Couldn't find any detections";
    top_shape[2] = 1;
    top[0]->Reshape(top_shape);
    caffe_set<Dtype>(top[0]->count(), -1, top[0]->mutable_cpu_data());
    return;
  }
  top[0]->Reshape(top_shape);
  Dtype* top_data = top[0]->mutable_cpu_data();

  int count = 0;
  boost::filesystem::path output_directory(output_directory_);
  for (int i = 0; i < num; ++i) {  //For each test image
    const map<int, vector<float> >& conf_scores = all_conf_scores[i]; // each should be #class by #prediction
    // Added on April 7th 2017
    const map<int, vector<float> >& age_scores = all_age_scores[i]; // Added by Dong Liu for MTL, each should be #prediction_id by num_classes
    //Below uncommented on March 10th 2017
    const map<int, vector<float> >& gender_scores = all_gender_scores[i]; //Added by Dong Liu for MTL, each should be # prediction_id by num_classes

    const LabelBBox& decode_bboxes = all_decode_bboxes[i];
    for (map<int, vector<int> >::iterator it = all_indices[i].begin();
         it != all_indices[i].end(); ++it) {
      int label = it->first;
      if (conf_scores.find(label) == conf_scores.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find confidence predictions for " << label;
        continue;
      }
      const vector<float>& scores = conf_scores.find(label)->second;
      int loc_label = share_location_ ? -1 : label;
      if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for " << loc_label;
        continue;
      }
      const vector<NormalizedBBox>& bboxes =
          decode_bboxes.find(loc_label)->second;
      vector<int>& indices = it->second;
      if (need_save_) {
        CHECK(label_to_name_.find(label) != label_to_name_.end())
          << "Cannot find label: " << label << " in the label map.";
        CHECK_LT(name_count_, names_.size());
      }
      for (int j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        const vector<float>& ascores = age_scores.find(idx)->second; //Added by Dong Liu for MTL
        // Below uncommented by Dong Liu on March 10th 2017
        const vector<float>& gscores = gender_scores.find(idx)->second; //Added by Dong Liu for MTL

        top_data[count * 11] = i;
        top_data[count * 11 + 1] = label;
        top_data[count * 11 + 2] = scores[idx];
        NormalizedBBox clip_bbox;
        ClipBBox(bboxes[idx], &clip_bbox);
        top_data[count * 11 + 3] = clip_bbox.xmin();
        top_data[count * 11 + 4] = clip_bbox.ymin();
        top_data[count * 11 + 5] = clip_bbox.xmax();
        top_data[count * 11 + 6] = clip_bbox.ymax();
        top_data[count * 11 + 7] = getIndexOfLargestElement(ascores, ascores.size()); //Added on April 7th 2017
        top_data[count * 11 + 8] = *(std::max_element(ascores.begin()+1, ascores.end())); //Added on April 7th 2017
        top_data[count * 11 + 9] = getIndexOfLargestElement(gscores, gscores.size()) ; //uncommented by Dong Liu on March 10th 2017
        top_data[count * 11 + 10] = *(std::max_element(gscores.begin()+1, gscores.end())); // uncommented by Dong Liu on March 10th 2017  added +1 on March 10th 2017
        // top_data[count * 9 + 7] = 0; commented on March 10th 2017
        // top_data[count * 9 + 8] = 0; commnted on March 10th 2017 


        if (need_save_) {
          NormalizedBBox scale_bbox;
          ScaleBBox(clip_bbox, sizes_[name_count_].first,
                    sizes_[name_count_].second, &scale_bbox);
          float score = top_data[count * 11 + 2]; //change 7 into 11
          int age = top_data[count * 11 + 7]; // added by Dong Liu for MTL
          float ascore = top_data[count * 11 + 8]; // Added by Dong Liu for MTL
          int gender = top_data[count * 11 + 9]; //Added By Dong Liu for MTL
          float gscore = top_data[count * 11 + 10]; // Added By Dong Liu for MTL

          float xmin = scale_bbox.xmin();
          float ymin = scale_bbox.ymin();
          float xmax = scale_bbox.xmax();
          float ymax = scale_bbox.ymax();
          ptree pt_xmin, pt_ymin, pt_width, pt_height;
          pt_xmin.put<float>("", round(xmin * 100) / 100.);
          pt_ymin.put<float>("", round(ymin * 100) / 100.);
          pt_width.put<float>("", round((xmax - xmin) * 100) / 100.);
          pt_height.put<float>("", round((ymax - ymin) * 100) / 100.);

          ptree cur_bbox;
          cur_bbox.push_back(std::make_pair("", pt_xmin));
          cur_bbox.push_back(std::make_pair("", pt_ymin));
          cur_bbox.push_back(std::make_pair("", pt_width));
          cur_bbox.push_back(std::make_pair("", pt_height));

          ptree cur_det;
          cur_det.put("image_id", names_[name_count_]);
          if (output_format_ == "ILSVRC") {
            cur_det.put<int>("category_id", label);
          } else {
            cur_det.put("category_id", label_to_name_[label].c_str());
          }
          cur_det.add_child("bbox", cur_bbox);
          cur_det.put<float>("score", score);
          cur_det.put<int>("age", age); //Added by Dong Liu for MTL
          cur_det.put<float>("ascore", ascore); // Added By Dong Liu for MTL
          cur_det.put<int>("gender", gender); //Added by Dong Liu for MTL
          cur_det.put<float>("gscore", gscore); //Added by Dong Liu for MTL

          detections_.push_back(std::make_pair("", cur_det));
        }
        ++count;
      } /// for each matched bounding box of a certain label
    } // for each label of the matched boxes in an image

    if (need_save_) {
      ++name_count_;
      if (name_count_ % num_test_image_ == 0) {
        if (output_format_ == "VOC") {
          map<string, std::ofstream*> outfiles;
          for (int c = 0; c < num_classes_; ++c) {
            if (c == background_label_id_) {
              continue;
            }
            string label_name = label_to_name_[c];
            boost::filesystem::path file(
                output_name_prefix_ + label_name + ".txt");
            boost::filesystem::path out_file = output_directory / file;
            outfiles[label_name] = new std::ofstream(out_file.string().c_str(),
                std::ofstream::out);
          }
          BOOST_FOREACH(ptree::value_type &det, detections_.get_child("")) {
            ptree pt = det.second;
            string label_name = pt.get<string>("category_id");
            if (outfiles.find(label_name) == outfiles.end()) {
              std::cout << "Cannot find " << label_name << std::endl;
              continue;
            }
            string image_name = pt.get<string>("image_id");
            float score = pt.get<float>("score");
            int age = pt.get<int>("age"); // added by Dong Liu for MTL
            float ascore = pt.get<float>("ascore"); //Added by Dong Liu for MTL
            int gender = pt.get<int>("gender"); // Added by Dong Liu for MTL
            float gscore = pt.get<float>("gscore"); // Added by Dong Liu for MTL

            vector<int> bbox;
            BOOST_FOREACH(ptree::value_type &elem, pt.get_child("bbox")) {
              bbox.push_back(static_cast<int>(elem.second.get_value<float>()));
            }
            *(outfiles[label_name]) << image_name;
            *(outfiles[label_name]) << " " << score;
            *(outfiles[label_name]) << " " << bbox[0] << " " << bbox[1];
            *(outfiles[label_name]) << " " << bbox[0] + bbox[2];
            *(outfiles[label_name]) << " " << bbox[1] + bbox[3];
            *(outfiles[label_name]) << " " << age << " " << ascore;
            *(outfiles[label_name]) << " " <<  gender << " " << gscore;
            *(outfiles[label_name]) << " " ;
            *(outfiles[label_name]) << std::endl;
          }
          for (int c = 0; c < num_classes_; ++c) {
            if (c == background_label_id_) {
              continue;
            }
            string label_name = label_to_name_[c];
            outfiles[label_name]->flush();
            outfiles[label_name]->close();
            delete outfiles[label_name];
          }
        } else if (output_format_ == "COCO") {
          boost::filesystem::path output_directory(output_directory_);
          boost::filesystem::path file(output_name_prefix_ + ".json");
          boost::filesystem::path out_file = output_directory / file;
          std::ofstream outfile;
          outfile.open(out_file.string().c_str(), std::ofstream::out);

          boost::regex exp("\"(null|true|false|-?[0-9]+(\\.[0-9]+)?)\"");
          ptree output;
          output.add_child("detections", detections_);
          std::stringstream ss;
          write_json(ss, output);
          std::string rv = boost::regex_replace(ss.str(), exp, "$1");
          outfile << rv.substr(rv.find("["), rv.rfind("]") - rv.find("["))
              << std::endl << "]" << std::endl;
        } else if (output_format_ == "ILSVRC") {
          boost::filesystem::path output_directory(output_directory_);
          boost::filesystem::path file(output_name_prefix_ + ".txt");
          boost::filesystem::path out_file = output_directory / file;
          std::ofstream outfile;
          outfile.open(out_file.string().c_str(), std::ofstream::out);

          BOOST_FOREACH(ptree::value_type &det, detections_.get_child("")) {
            ptree pt = det.second;
            int label = pt.get<int>("category_id");
            string image_name = pt.get<string>("image_id");
            float score = pt.get<float>("score");
            vector<int> bbox;
            BOOST_FOREACH(ptree::value_type &elem, pt.get_child("bbox")) {
              bbox.push_back(static_cast<int>(elem.second.get_value<float>()));
            }
            outfile << image_name << " " << label << " " << score;
            outfile << " " << bbox[0] << " " << bbox[1];
            outfile << " " << bbox[0] + bbox[2];
            outfile << " " << bbox[1] + bbox[3];
            outfile << std::endl;
          }
        }
        name_count_ = 0;
        detections_.clear();
      }
    }// if need_save
  } // for each test image

  
  /* Below commented on March 10th 2017
  //******************************************** Below added by Dong Liu : Perform NMS on gender detection

  //*************************************  Below Added by Dong Liu: Perform NMS on gender bounding boxes *******************************************
   int gender_num_kept = 0;
   vector<map<int, vector<int> > > gender_all_indices; // commented by Dong Liu on March 10th 2017
   for (int i = 0; i < num; ++i) {
       const LabelBBox& decode_bboxes = all_decode_bboxes[i];
       const map<int, vector<float> >& gender_scores = all_gender_scores[i];
       map<int, vector<int> > gender_indices;
       int gender_num_det = 0;
       for (int c = 0; c < num_gender_classes_; ++c) {
         if (c == gender_background_label_id_) {
           // Ignore background class.
           continue;
         }
         if (gender_scores.find(c) == gender_scores.end()) {
           // Something bad happened if there are no predictions for current label.
           LOG(FATAL) << "Could not find gender confidence predictions for label " << c;
         }
         const vector<float>& scores = gender_scores.find(c)->second;
         int label = share_location_ ? -1 : c;
         if (decode_bboxes.find(label) == decode_bboxes.end()) {
           // Something bad happened if there are no predictions for current label.
           LOG(FATAL) << "Could not find location predictions for label " << label;
           continue;
         }
         const vector<NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;
         ApplyNMSFast(bboxes, scores, confidence_threshold_, nms_threshold_,
             top_k_, &(gender_indices[c]));
         gender_num_det += gender_indices[c].size();
       } // collect indices after nms

       if (keep_top_k_ > -1 && gender_num_det > keep_top_k_) {
         vector<pair<float, pair<int, int> > > score_index_pairs;
         for (map<int, vector<int> >::iterator it = gender_indices.begin();
              it != gender_indices.end(); ++it) {
           int label = it->first;
           const vector<int>& label_indices = it->second;
           if (gender_scores.find(label) == gender_scores.end()) {
             // Something bad happened for current label.
             LOG(FATAL) << "Could not find location predictions for " << label;
             continue;
           }
           const vector<float>& scores = gender_scores.find(label)->second;
           for (int j = 0; j < label_indices.size(); ++j) {
             int idx = label_indices[j];
             CHECK_LT(idx, scores.size());
             score_index_pairs.push_back(std::make_pair(
                     scores[idx], std::make_pair(label, idx)));
           }
         }
         // Keep top k results per image.
         std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                   SortScorePairDescend<pair<int, int> >);
         score_index_pairs.resize(keep_top_k_);
         // Store the new indices.
         map<int, vector<int> > new_indices;
         for (int j = 0; j < score_index_pairs.size(); ++j) {
           int label = score_index_pairs[j].second.first;
           int idx = score_index_pairs[j].second.second;
           new_indices[label].push_back(idx);
         }
         gender_all_indices.push_back(new_indices);
         gender_num_kept += keep_top_k_;
       } else {
         gender_all_indices.push_back(gender_indices);
         gender_num_kept += gender_num_det;
       }
     } // for each image


    vector<int> gender_top_shape(2, 1);
    gender_top_shape.push_back(gender_num_kept);
    gender_top_shape.push_back(9);
    if (gender_num_kept == 0) {
       LOG(INFO) << "Couldn't find any detections for gender";
       gender_top_shape[2] = 1;
       top[1]->Reshape(gender_top_shape);
       caffe_set<Dtype>(top[1]->count(), -1, top[1]->mutable_cpu_data());
       return;
    }
    top[1]->Reshape(gender_top_shape);
    Dtype* gender_top_data = top[1]->mutable_cpu_data(); // Added by Dong Liu for MTL

   // *************************************** Above added by Dong Liu : Perform NMS on gender detection ***********************************************

  int gender_count = 0;
  boost::filesystem::path gender_output_directory(output_directory_);
  for (int i = 0; i < num; ++i) {  //For each test image
      //const map<int, vector<float> >& conf_scores = all_conf_scores[i]; // each should be #class by #prediction
      //const map<int, vector<float> >& age_scores = all_age_scores[i]; // Added by Dong Liu for MTL, each should be #prediction_id by num_classes
      const map<int, vector<float> >& gender_scores = all_gender_scores[i]; //Added by Dong Liu for MTL, each should be # prediction_id by num_classes

      const LabelBBox& decode_bboxes = all_decode_bboxes[i];
      for (map<int, vector<int> >::iterator it = gender_all_indices[i].begin();
           it != gender_all_indices[i].end(); ++it) {
        int label = it->first;
        if (gender_scores.find(label) == gender_scores.end()) {
          // Something bad happened if there are no predictions for current label.
          LOG(FATAL) << "Could not find gender confidence predictions for " << label;
          continue;
        }
        const vector<float>& scores = gender_scores.find(label)->second;
        int loc_label = share_location_ ? -1 : label;
        if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
          // Something bad happened if there are no predictions for current label.
          LOG(FATAL) << "Could not find location predictions for " << loc_label;
          continue;
        }
        const vector<NormalizedBBox>& bboxes =
            decode_bboxes.find(loc_label)->second;
        vector<int>& indices = it->second;
        //if (need_save_) {
        //  CHECK(label_to_name_.find(label) != label_to_name_.end())
        //    << "Cannot find label: " << label << " in the label map.";
        //  CHECK_LT(name_count_, names_.size());
        //}
        for (int j = 0; j < indices.size(); ++j) {
          int idx = indices[j];
          //const vector<float>& ascores = age_scores.find(idx)->second; //Added by Dong Liu for MTL
          //const vector<float>& gscores = gender_scores.find(idx)->second; //Added by Dong Liu for MTL

          gender_top_data[gender_count * 9] = i;
          gender_top_data[gender_count * 9 + 1] = 0;
          gender_top_data[gender_count * 9 + 2] = 0; //scores[idx];
          NormalizedBBox clip_bbox;
          ClipBBox(bboxes[idx], &clip_bbox);
          gender_top_data[gender_count * 9 + 3] = clip_bbox.xmin();
          gender_top_data[gender_count * 9 + 4] = clip_bbox.ymin();
          gender_top_data[gender_count * 9 + 5] = clip_bbox.xmax();
          gender_top_data[gender_count * 9 + 6] = clip_bbox.ymax();
          //top_data[count * 11 + 7] = getIndexOfLargestElement(ascores, ascores.size());
          //top_data[count * 11 + 8] = *(std::max_element(ascores.begin(), ascores.end()));
          gender_top_data[gender_count * 9 + 7] = label; //getIndexOfLargestElement(gscores, gscores.size()) ;
          gender_top_data[gender_count * 9 + 8] = scores[idx]; //*(std::max_element(gscores.begin(), gscores.end()));


          if (need_save_) {
            NormalizedBBox scale_bbox;
            ScaleBBox(clip_bbox, sizes_[gender_name_count_].first,
                      sizes_[gender_name_count_].second, &scale_bbox);
            float score = gender_top_data[gender_count * 9 + 1]; //change 7 into 11
            //int age = top_data[count * 11 + 7]; // added by Dong Liu for MTL
            //float ascore = top_data[count * 11 + 8]; // Added by Dong Liu for MTL
            int gender = gender_top_data[gender_count * 9 + 7]; //Added By Dong Liu for MTL
            float gscore = gender_top_data[gender_count * 9 + 8]; // Added By Dong Liu for MTL

            float xmin = scale_bbox.xmin();
            float ymin = scale_bbox.ymin();
            float xmax = scale_bbox.xmax();
            float ymax = scale_bbox.ymax();
            ptree pt_xmin, pt_ymin, pt_width, pt_height;
            pt_xmin.put<float>("", round(xmin * 100) / 100.);
            pt_ymin.put<float>("", round(ymin * 100) / 100.);
            pt_width.put<float>("", round((xmax - xmin) * 100) / 100.);
            pt_height.put<float>("", round((ymax - ymin) * 100) / 100.);

            ptree cur_bbox;
            cur_bbox.push_back(std::make_pair("", pt_xmin));
            cur_bbox.push_back(std::make_pair("", pt_ymin));
            cur_bbox.push_back(std::make_pair("", pt_width));
            cur_bbox.push_back(std::make_pair("", pt_height));

            ptree cur_det;

            string label_name;
            if(label == 1)
            	label_name = "female";
            else if(label == 2)
            	label_name = "male";

            cur_det.put("image_id", names_[gender_name_count_]);
            if (output_format_ == "ILSVRC") {
              cur_det.put<int>("category_id", label);
            } else {
              //cur_det.put("category_id", label_to_name_[label].c_str());
              cur_det.put("category_id", label_name.c_str());
            }
            cur_det.add_child("bbox", cur_bbox);
            cur_det.put<float>("score", score);
            //cur_det.put<int>("age", age); //Added by Dong Liu for MTL
            //cur_det.put<float>("ascore", ascore); // Added By Dong Liu for MTL
            cur_det.put<int>("gender", gender); //Added by Dong Liu for MTL
            cur_det.put<float>("gscore", gscore); //Added by Dong Liu for MTL

            gender_detections_.push_back(std::make_pair("", cur_det));
          }
          ++gender_count;
        } /// for each matched bounding box of a certain label
      } // for each label of the matched boxes in an image

      if (need_save_) {
        ++gender_name_count_;
        if (gender_name_count_ % num_test_image_ == 0) {
          if (output_format_ == "VOC") {
            map<string, std::ofstream*> outfiles;
            for (int c = 0; c < num_gender_classes_; ++c) {
              if (c == gender_background_label_id_) {
                continue;
              }
              string label_name; // = label_to_name_[c];

              if(c==1)
            	 label_name = "female";
              else if(c==2)
            	 label_name = "male";

              boost::filesystem::path file(
                  output_name_prefix_ + label_name + ".txt");
              boost::filesystem::path out_file = gender_output_directory / file;
              outfiles[label_name] = new std::ofstream(out_file.string().c_str(),
                  std::ofstream::out);
            }

            BOOST_FOREACH(ptree::value_type &det, gender_detections_.get_child("")) {
              ptree pt = det.second;
              string label_name = pt.get<string>("category_id");
              if (outfiles.find(label_name) == outfiles.end()) {
                std::cout << "Cannot find " << label_name << std::endl;
                continue;
              }
              string image_name = pt.get<string>("image_id");
              float score = pt.get<float>("score");
              //int age = pt.get<int>("age"); // added by Dong Liu for MTL
              //float ascore = pt.get<float>("ascore"); //Added by Dong Liu for MTL
              int gender = pt.get<int>("gender"); // Added by Dong Liu for MTL
              float gscore = pt.get<float>("gscore"); // Added by Dong Liu for MTL

              vector<int> bbox;
              BOOST_FOREACH(ptree::value_type &elem, pt.get_child("bbox")) {
                bbox.push_back(static_cast<int>(elem.second.get_value<float>()));
              }
              *(outfiles[label_name]) << image_name;
              *(outfiles[label_name]) << " " << score;
              *(outfiles[label_name]) << " " << bbox[0] << " " << bbox[1];
              *(outfiles[label_name]) << " " << bbox[0] + bbox[2];
              *(outfiles[label_name]) << " " << bbox[1] + bbox[3];
              //*(outfiles[label_name]) << " " << age << " " << ascore;
              *(outfiles[label_name]) << " " <<  gender << " " << gscore;
              *(outfiles[label_name]) << " " ;
              *(outfiles[label_name]) << std::endl;
            }
            for (int c = 0; c < num_gender_classes_; ++c) {
              if (c == gender_background_label_id_) {
                continue;
              }
              //string label_name = label_to_name_[c];

              string label_name;
              if(c == 1)
            	 label_name = "female";
              else if(c == 2)
            	 label_name = "male";

              outfiles[label_name]->flush();
              outfiles[label_name]->close();
              delete outfiles[label_name];
            }
          } else if (output_format_ == "COCO") {
            boost::filesystem::path output_directory(output_directory_);
            boost::filesystem::path file(output_name_prefix_ + ".json");
            boost::filesystem::path out_file = output_directory / file;
            std::ofstream outfile;
            outfile.open(out_file.string().c_str(), std::ofstream::out);

            boost::regex exp("\"(null|true|false|-?[0-9]+(\\.[0-9]+)?)\"");
            ptree output;
            output.add_child("detections", detections_);
            std::stringstream ss;
            write_json(ss, output);
            std::string rv = boost::regex_replace(ss.str(), exp, "$1");
            outfile << rv.substr(rv.find("["), rv.rfind("]") - rv.find("["))
                << std::endl << "]" << std::endl;
          } else if (output_format_ == "ILSVRC") {
            boost::filesystem::path output_directory(output_directory_);
            boost::filesystem::path file(output_name_prefix_ + ".txt");
            boost::filesystem::path out_file = output_directory / file;
            std::ofstream outfile;
            outfile.open(out_file.string().c_str(), std::ofstream::out);

            BOOST_FOREACH(ptree::value_type &det, detections_.get_child("")) {
              ptree pt = det.second;
              int label = pt.get<int>("category_id");
              string image_name = pt.get<string>("image_id");
              float score = pt.get<float>("score");
              vector<int> bbox;
              BOOST_FOREACH(ptree::value_type &elem, pt.get_child("bbox")) {
                bbox.push_back(static_cast<int>(elem.second.get_value<float>()));
              }
              outfile << image_name << " " << label << " " << score;
              outfile << " " << bbox[0] << " " << bbox[1];
              outfile << " " << bbox[0] + bbox[2];
              outfile << " " << bbox[1] + bbox[3];
              outfile << std::endl;
            }
          }
          gender_name_count_ = 0;
          gender_detections_.clear();
        }
      }// if need_save
    } // for each test image 
    
    above commented on March 10th 2017*/ 


/*if (visualize_) {
#ifdef USE_OPENCV
    vector<cv::Mat> cv_imgs;
    this->data_transformer_->TransformInv(bottom[4], &cv_imgs);
    vector<cv::Scalar> colors = GetColors(label_to_display_name_.size());
    VisualizeBBox(cv_imgs, top[0], visualize_threshold_, colors,
        label_to_display_name_);
#endif  // USE_OPENCV
  }*/ //commneted by Dong Liu on March 13th 2017
  
  if (visualize_) {
    #ifdef USE_OPENCV
       if(need_save_frame_) {
         vector<cv::Mat> cv_imgs;
         this->data_transformer_->TransformInv(bottom[5], &cv_imgs);
         //vector<cv::Scalar> colors = GetColors(label_to_display_name_.size()); //commented by Dong Liu on March 13th 2017
         vector<cv::Scalar> colors = GetColors(3); // added by Dong Liu on March 13th 2017 
         VisualizeBBox(cv_imgs, top[0], visualize_threshold_, colors,
                       label_to_display_name_, frame_output_directory_, visualize_count_* batch_size_);
         visualize_count_++;
       } else {
         vector<cv::Mat> cv_imgs;
         this->data_transformer_->TransformInv(bottom[5], &cv_imgs);
         vector<cv::Scalar> colors = GetColors(label_to_display_name_.size());
         VisualizeBBox(cv_imgs, top[0], visualize_threshold_, colors,
         label_to_display_name_);
      }
    #endif  // USE_OPENCV
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DetectionOutputLayer);

}  // namespace caffe
