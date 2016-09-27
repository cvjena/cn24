/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
/**
 * @file makeTensorStream.cpp
 * @brief Tool to import datasets
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#include <iostream>
#include <fstream>

#include <cn24.h>

int main ( int argc, char** argv ) {
  if ( argc < 8 ) {
    LOGERROR << "USAGE: " << argv[0] << " <dataset config file> <image list file> <image directory> <label list file> <label directory> <output file> <true/false for direct RGB of labels>";
    LOGEND;
    return -1;
  }
  
  Conv::System::Init(3);

  // Capture command line arguments
  std::string directRGB ( argv[7] );
  std::string output_fname ( argv[6] );
  std::string label_directory ( argv[5] );
  std::string label_list_fname ( argv[4] );
  std::string image_directory ( argv[3] );
  std::string image_list_fname ( argv[2] );
  std::string dataset_config_fname ( argv[1] );

  if(image_directory.back() != '/')
    image_directory += "/";

  if(label_directory.back() != '/')
    label_directory += "/";

  // Open dataset configuration files
  std::ifstream dataset_config_file ( dataset_config_fname,std::ios::in );

  if ( !dataset_config_file.good() ) {
    FATAL ( "Cannot open dataset configuration file!" );
  }

  LOGINFO << "Loading dataset";
  // Load dataset
  Conv::ClassManager class_manager;
  Conv::TensorStreamDataset* dataset = Conv::TensorStreamDataset::CreateFromConfiguration ( dataset_config_file, true, Conv::LOAD_BOTH, &class_manager);
  UNREFERENCED_PARAMETER(dataset);

  unsigned int number_of_classes = class_manager.GetMaxClassId() + 1;
  // arrays to store class colors in an easy to index way
  Conv::datum* cr = new Conv::datum[number_of_classes];
  Conv::datum* cg = new Conv::datum[number_of_classes];
  Conv::datum* cb = new Conv::datum[number_of_classes];

  for(Conv::ClassManager::const_iterator it = class_manager.begin(); it != class_manager.end(); it++) {
    const unsigned int c = it->second.id;
    const unsigned int class_color = it->second.color;
    cr[c] = DATUM_FROM_UCHAR ( ( class_color >> 16 ) & 0xFF );
    cg[c] = DATUM_FROM_UCHAR ( ( class_color >> 8 ) & 0xFF );
    cb[c] = DATUM_FROM_UCHAR ( class_color & 0xFF );
  }

  // Open file lists
  std::ifstream image_list_file ( image_list_fname, std::ios::in );

  if ( !image_list_file.good() ) {
    FATAL ( "Cannot open image list file!" );
  }

  std::ifstream label_list_file ( label_list_fname, std::ios::in );

  if ( !label_list_file.good() ) {
    FATAL ( "Cannot open label list file!" );
  }

  // Open output file
  std::ofstream output_file ( output_fname, std::ios::out | std::ios::binary );

  if ( !output_file.good() ) {
    FATAL ( "Cannot open output file!" );
  }

  // Iterate through lists of images and labels
  while ( !image_list_file.eof() ) {
    std::string image_fname;
    std::string label_fname;
    std::getline ( image_list_file, image_fname );
    std::getline ( label_list_file, label_fname );

    if ( image_fname.length() < 5 || label_fname.length() < 5 )
      break;

    LOGINFO << "Importing files " << image_fname << " and " << label_fname << "...";
    Conv::Tensor image_tensor ( image_directory + image_fname );
    Conv::Tensor label_rgb_tensor ( label_directory + label_fname );

    if ( image_tensor.width() != label_rgb_tensor.width() ||
         image_tensor.height() != label_rgb_tensor.height() ) {
      LOGERROR << "Dimensions don't match, skipping file!";
      continue;
    }
 
    int label_tensor_width = number_of_classes; 
    if(directRGB == "true") {
      label_tensor_width = 3;
    }
	
    Conv::Tensor label_tensor ( 1, label_rgb_tensor.width(), label_rgb_tensor.height(), label_tensor_width);

    if(directRGB == "true") {
      // no classes - interpret the label tensor input as the output (no class/color mapping)
       for ( unsigned int y = 0; y < label_rgb_tensor.height(); y++ ) {
        for ( unsigned int x = 0; x < label_rgb_tensor.width(); x++ ) {     
          *label_tensor.data_ptr ( x,y,0,0 ) = *label_rgb_tensor.data_ptr_const ( x,y,0,0 );
          *label_tensor.data_ptr ( x,y,1,0 ) = *label_rgb_tensor.data_ptr_const ( x,y,1,0 );
          *label_tensor.data_ptr ( x,y,2,0 ) = *label_rgb_tensor.data_ptr_const ( x,y,2,0 );
        }
      }
    } else if(number_of_classes == 1) {
      // 1 class - convert RGB images into multi-channel label tensors
      const unsigned int foreground_color = class_manager.begin()->second.color;
      const Conv::datum fr = DATUM_FROM_UCHAR ( ( foreground_color >> 16 ) & 0xFF ),
                        fg = DATUM_FROM_UCHAR ( ( foreground_color >> 8 ) & 0xFF ),
                        fb = DATUM_FROM_UCHAR ( foreground_color & 0xFF );

      for ( unsigned int y = 0; y < label_rgb_tensor.height(); y++ ) {
        for ( unsigned int x = 0; x < label_rgb_tensor.width(); x++ ) {
          Conv::datum lr, lg, lb;

          if ( label_rgb_tensor.maps() == 3 ) {
            lr = *label_rgb_tensor.data_ptr_const ( x,y,0,0 );
            lg = *label_rgb_tensor.data_ptr_const ( x,y,1,0 );
            lb = *label_rgb_tensor.data_ptr_const ( x,y,2,0 );
          } else if ( label_rgb_tensor.maps() == 1 ) {
            lr = *label_rgb_tensor.data_ptr_const ( x,y,0,0 );
            lg = lr;
            lb = lr;
          } else {
            FATAL ( "Unsupported input channel count!" );
          }

          const Conv::datum class1_diff = std::sqrt ( ( lr - fr ) * ( lr - fr )
                                          + ( lg - fg ) * ( lg - fg )
                                          + ( lb - fb ) * ( lb - fb ) ) / std::sqrt ( 3.0 );
          const Conv::datum val = 1.0 - 2.0 * class1_diff;
          *label_tensor.data_ptr ( x,y,0,0 ) = val;
        }
      }
    } else {
      // any number of other classes      
      label_tensor.Clear ( 0.0 );

      for ( unsigned int y = 0; y < label_rgb_tensor.height(); y++ ) {
        for ( unsigned int x = 0; x < label_rgb_tensor.width(); x++ ) {
          Conv::datum lr, lg, lb;

          if ( label_rgb_tensor.maps() == 3 ) {
            lr = *label_rgb_tensor.data_ptr_const ( x,y,0,0 );
            lg = *label_rgb_tensor.data_ptr_const ( x,y,1,0 );
            lb = *label_rgb_tensor.data_ptr_const ( x,y,2,0 );
          } else if ( label_rgb_tensor.maps() == 1 ) {
            lr = *label_rgb_tensor.data_ptr_const ( x,y,0,0 );
            lg = lr;
            lb = lr;
          } else {
            FATAL ( "Unsupported input channel count!" );
          }

          for ( unsigned int c = 0; c <number_of_classes; c++ ) {
            if(lr == cr[c] && lg == cg[c] && lb == cb[c])
              *label_tensor.data_ptr ( x,y,c,0 ) = 1.0;
          }
        }
      }
    } // end if

    image_tensor.Serialize ( output_file );
    label_tensor.Serialize ( output_file );
  }

  LOGEND;
}
