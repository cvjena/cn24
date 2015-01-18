/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file convertKITTI.cpp
 * \brief Program to convert KITTI data to Tensor format.
 *
 * You can find the dataset at http://yann.lecun.com/exdb/mnist/
 *
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#include <iostream>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <cstdint>

#include <cn24.h>

void convertCategory (Conv::KITTIData& data, Conv::KITTICategory category);
void convertCategoryStreamlined (Conv::KITTIData& data,
                                 Conv::KITTICategory category);

int main (int argc, char* argv[]) {
  if (argc < 2) {
    LOGERROR << "USAGE: " << argv[0] << " <KITTI data folder> ";
    LOGEND;
    return -1;
  }

  Conv::KITTIData data (argv[1]);

  /*convertCategory (data, Conv::KITTI_UM);
  convertCategory (data, Conv::KITTI_UMM);
  convertCategory (data, Conv::KITTI_UU);*/

  convertCategoryStreamlined (data, Conv::KITTI_UM);
  convertCategoryStreamlined (data, Conv::KITTI_UMM);
  convertCategoryStreamlined (data, Conv::KITTI_UU);

  LOGEND;

}

void convertCategory (Conv::KITTIData& data, Conv::KITTICategory category) {
  int first = 0;
  int last = 0;
  std::string cat_str;
  switch (category) {
    case Conv::KITTI_UM:
      first = 1; last = 94; cat_str = "UM"; break;
    case Conv::KITTI_UMM:
      first = 0; last = 95; cat_str = "UMM"; break;
    case Conv::KITTI_UU:
      first = 0; last = 97; cat_str = "UU"; break;
    case Conv::KITTI_URBAN:
      FATAL ("URBAN not supported right now!"); break;
  }

  std::ofstream tensor_stream ("output_" + cat_str + ".Tensor",
                               std::ios::binary | std::ios::out);
  std::ofstream gt_stream ("gt_road_" + cat_str + ".Tensor",
                           std::ios::binary | std::ios::out);

  for (int i = 0; i <= (last - first); i++) {
    std::string file_name = data.getImage (category, first + i);
    std::ifstream image_stream (file_name, std::ios::binary | std::ios::in);
    if (!image_stream.good()) {
      FATAL ("Cannot open " << file_name);
    }
    Conv::Tensor image;
    Conv::PNGLoader::LoadFromStream (image_stream, image);
/*    for (unsigned int i = 0; i < image.elements(); i++) {
      image[i] -= 0.5;
      image[i] *= 2.0;
    }*/
    image.Serialize (tensor_stream);
    image_stream.close();

    std::string gt_file_name = data.getRoadGroundtruth (category, first + i);
    std::ifstream gt_image_stream (gt_file_name, std::ios::binary | std::ios::in);
    if (!gt_image_stream.good()) {
      FATAL ("Cannot open " << gt_file_name);
    }
    Conv::Tensor gt;
    Conv::PNGLoader::LoadFromStream (gt_image_stream, gt);
    for (unsigned int i = 0; i < gt.elements(); i++) {
      gt[i] -= 0.5;
      gt[i] *= 2.0;
    }
    gt.Serialize (gt_stream);
    gt_stream.close();
  }

  if (category == Conv::KITTI_UM) {
    std::ofstream gtl_stream ("gt_lane_" + cat_str + ".Tensor",
                              std::ios::binary | std::ios::out);

    for (int i = 0; i <= (last - first); i++) {
      std::string gtl_file_name = data.getLaneGroundtruth (category, first + i);
      std::ifstream gtl_image_stream (gtl_file_name, std::ios::binary | std::ios::in);
      if (!gtl_image_stream.good()) {
        FATAL ("Cannot open " << gtl_file_name);
      }
      Conv::Tensor gtl;
      Conv::PNGLoader::LoadFromStream (gtl_image_stream, gtl);
      for (unsigned int i = 0; i < gtl.elements(); i++) {
        gtl[i] -= 0.5;
        gtl[i] *= 2.0;
      }
      gtl.Serialize (gtl_stream);
      gtl_stream.close();
    }
  }

  LOGINFO << "Category " << cat_str << " converted" << std::flush;
}

void convertCategoryStreamlined (Conv::KITTIData& data,
                                 Conv::KITTICategory category) {
  unsigned int resx = 1226;
  unsigned int resy = 370;
  unsigned int first = 0;
  unsigned int last = 0;
  std::string cat_str;
  switch (category) {
    case Conv::KITTI_UM:
      first = 1; last = 94; cat_str = "UM"; break;
    case Conv::KITTI_UMM:
      first = 0; last = 95; cat_str = "UMM"; break;
    case Conv::KITTI_UU:
      first = 0; last = 97; cat_str = "UU"; break;
    case Conv::KITTI_URBAN:
      FATAL ("URBAN not supported right now!"); break;
  }

  std::ofstream tensor_stream (cat_str + "_road.Tensor",
                               std::ios::binary | std::ios::out);
  Conv::Tensor images (1 + last - first, resx, resy, 3);
  Conv::Tensor gts (1 + last - first, resx, resy, 1);

  for (unsigned int i = 0; i <= (last - first); i++) {
    std::string file_name = data.getImage (category, first + i);
    std::ifstream image_stream (file_name, std::ios::binary | std::ios::in);
    if (!image_stream.good()) {
      FATAL ("Cannot open " << file_name);
    }
    Conv::Tensor image (1, resx, resy, 3);
    Conv::Tensor image_st (1, resx, resy, 3);
    Conv::PNGLoader::LoadFromStream (image_stream, image);
    /*for (unsigned int i = 0; i < image.elements(); i++) {
      image[i] -= 0.5;
      image[i] *= 2.0;
    }*/
    for (unsigned int y = 0; y < resy; y++) {
      std::memcpy (image_st.data_ptr (0, y, 0), image.data_ptr (0, y, 0),
                   sizeof (Conv::datum) * resx);
      std::memcpy (image_st.data_ptr (0, y, 1), image.data_ptr (0, y, 1),
                   sizeof (Conv::datum) * resx);
      std::memcpy (image_st.data_ptr (0, y, 2), image.data_ptr (0, y, 2),
                   sizeof (Conv::datum) * resx);
    }


    std::string gt_file_name = data.getRoadGroundtruth (category, first + i);
    std::ifstream gt_image_stream (gt_file_name, std::ios::binary | std::ios::in);
    if (!gt_image_stream.good()) {
      FATAL ("Cannot open " << gt_file_name);
    }
    Conv::Tensor gt;
    Conv::Tensor gt_st(1, resx, resy, 1);
    Conv::PNGLoader::LoadFromStream (gt_image_stream, gt);
    for (unsigned int i = 0; i < gt.elements(); i++) {
      gt[i] -= 0.5;
      gt[i] *= 2.0;
    }
    for (unsigned int y = 0; y < resy; y++) {
      std::memcpy (gt_st.data_ptr (0, y, 0),
                   gt.data_ptr (0, y, 0),
                   sizeof (Conv::datum) * resx);
      
    }
    
    image_st.Serialize(tensor_stream);
    gt_st.Serialize(tensor_stream);
    
    image_stream.close();
    gt_image_stream.close();
  }


  // images.Serialize (tensor_stream);
  // gts.Serialize (tensor_stream);
  tensor_stream.close();

  if (category == Conv::KITTI_UM) {
    std::ofstream gtl_stream (cat_str + "_lane.Tensor",
                              std::ios::binary | std::ios::out);

    Conv::Tensor gtls (1 + last - first, resx, resy, 1);

    for (unsigned int i = 0; i <= (last - first); i++) {
      std::string gtl_file_name = data.getLaneGroundtruth (category, first + i);
      std::ifstream gtl_image_stream (gtl_file_name, std::ios::binary | std::ios::in);
      if (!gtl_image_stream.good()) {
        FATAL ("Cannot open " << gtl_file_name);
      }
      Conv::Tensor gtl;
      Conv::PNGLoader::LoadFromStream (gtl_image_stream, gtl);
      for (unsigned int i = 0; i < gtl.elements(); i++) {
        gtl[i] -= 0.5;
        gtl[i] *= 2.0;
      }
      for (unsigned int y = 0; y < resy; y++) {
        std::memcpy (gtls.data_ptr (0, y, 0, i),
                     gtl.data_ptr (0, y, 0),
                     sizeof (Conv::datum) * resx);
      }

      gtl_image_stream.close();
    }
    
    images.Serialize(gtl_stream);
    gtls.Serialize(gtl_stream);
    gtl_stream.close();
  }
}

