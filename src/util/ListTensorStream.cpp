/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifdef BUILD_BOOST
#include <boost/regex.hpp>
#else
#include <regex>
#endif

#include <iostream>
#include <fstream>
#include <cmath>

#include "ListTensorStream.h"

namespace Conv {
  std::size_t ListTensorStream::GetWidth(unsigned int index) {
		if(index < tensors_.size())
			return tensors_[index].width;
		else
			return 0;
  }
  
  std::size_t ListTensorStream::GetHeight(unsigned int index) {
		if(index < tensors_.size())
			return tensors_[index].height;
		else
			return 0;
  }
  
  std::size_t ListTensorStream::GetMaps(unsigned int index) {
		if(index < tensors_.size())
			return tensors_[index].maps;
		else
			return 0;
  }
  
  std::size_t ListTensorStream::GetSamples(unsigned int index) {
		if(index < tensors_.size())
			return tensors_[index].samples;
		else
			return 0;
  }
  
  unsigned int ListTensorStream::GetTensorCount() {
    return tensors_.size();
  }
  
  unsigned int ListTensorStream::LoadFile(std::string path) {
		std::string listtensor_regex = "list:(.*);(.*);(.*);(.*)";
#ifdef BUILD_BOOST
		boost::smatch listtensor_match;
		bool has_matches = boost::regex_match(path, listtensor_match, boost::regex(listtensor_regex, boost::regex::extended));
#else
		std::smatch listtensor_match;
		bool has_matches = std::regex_match(path, listtensor_match, std::regex(listtensor_regex, std::regex::extended));
#endif
		if(has_matches && listtensor_match.size() == 5) {
			std::string imagelist_path = listtensor_match[1];
			std::string images = listtensor_match[2];
			std::string labellist_path = listtensor_match[3];
			std::string labels = listtensor_match[4];
			
			return LoadFiles(imagelist_path, images, labellist_path, labels);
		} else {
			LOGERROR << "Not a valid list descriptor: " << path;
			LOGERROR << "Found " << listtensor_match.size() << " matches.";
			return 0;
		}
  }
  
  bool ListTensorStream::CopySample(const unsigned int source_index, const std::size_t source_sample, Conv::Tensor &target, const std::size_t target_sample, const bool scale) {
		if(source_index < tensors_.size()) {
			// Load tensor by filename
			Tensor rgb_tensor;
			if(!tensors_[source_index].ignore)
        rgb_tensor.LoadFromFile(tensors_[source_index].filename);
      else {
				target.Clear((datum) 0.0, target_sample);
				return true;
			}
			
			// TODO Consider validation vs. performance
			
			if(source_index % 2 && rgb_tensor.elements() > 0) {
				// Tensor has a label in it, colors need to be transformed
				unsigned int number_of_classes = class_manager_->GetMaxClassId() + 1;
				
				// arrays to store class colors in an easy to index way
				Conv::datum* cr = new Conv::datum[number_of_classes];
				Conv::datum* cg = new Conv::datum[number_of_classes];
				Conv::datum* cb = new Conv::datum[number_of_classes];

				for(ClassManager::const_iterator it = class_manager_->begin(); it != class_manager_->end(); it++) {
          const unsigned int c = it->second.id;
					const unsigned int class_color = it->second.color;
					cr[c] = DATUM_FROM_UCHAR ( ( class_color >> 16 ) & 0xFF );
					cg[c] = DATUM_FROM_UCHAR ( ( class_color >> 8 ) & 0xFF );
					cb[c] = DATUM_FROM_UCHAR ( class_color & 0xFF );
				}
				
				Conv::Tensor label_tensor ( 1, rgb_tensor.width(), rgb_tensor.height(), number_of_classes);

				if(number_of_classes == 1) {
					// 1 class - convert RGB images into multi-channel label tensors
					const unsigned int foreground_color = class_manager_->begin()->second.color;
					const Conv::datum fr = DATUM_FROM_UCHAR ( ( foreground_color >> 16 ) & 0xFF ),
														fg = DATUM_FROM_UCHAR ( ( foreground_color >> 8 ) & 0xFF ),
														fb = DATUM_FROM_UCHAR ( foreground_color & 0xFF );

					for ( unsigned int y = 0; y < rgb_tensor.height(); y++ ) {
						for ( unsigned int x = 0; x < rgb_tensor.width(); x++ ) {
							Conv::datum lr, lg, lb;

							if ( rgb_tensor.maps() == 3 ) {
								lr = *rgb_tensor.data_ptr_const ( x,y,0,0 );
								lg = *rgb_tensor.data_ptr_const ( x,y,1,0 );
								lb = *rgb_tensor.data_ptr_const ( x,y,2,0 );
							} else if ( rgb_tensor.maps() == 1 ) {
								lr = *rgb_tensor.data_ptr_const ( x,y,0,0 );
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

					for ( unsigned int y = 0; y < rgb_tensor.height(); y++ ) {
						for ( unsigned int x = 0; x < rgb_tensor.width(); x++ ) {
							Conv::datum lr, lg, lb;

							if ( rgb_tensor.maps() == 3 ) {
								lr = *rgb_tensor.data_ptr_const ( x,y,0,0 );
								lg = *rgb_tensor.data_ptr_const ( x,y,1,0 );
								lb = *rgb_tensor.data_ptr_const ( x,y,2,0 );
							} else if ( rgb_tensor.maps() == 1 ) {
								lr = *rgb_tensor.data_ptr_const ( x,y,0,0 );
								lg = lr;
								lb = lr;
							} else {
								FATAL ( "Unsupported input channel count!" );
							}

							for ( unsigned int c = 0; c < number_of_classes; c++ ) {
								if(lr == cr[c] && lg == cg[c] && lb == cb[c])
									*label_tensor.data_ptr ( x,y,c,0 ) = 1.0;
							}
						}
					}
				} // end if
				return Tensor::CopySample(label_tensor, source_sample, target, target_sample, false, scale);
				
			} else {
				// Tensor has an image in it, no transform needed
        if(rgb_tensor.maps() == 3) {
          // Tensor is already RGB
					return Tensor::CopySample(rgb_tensor, source_sample, target, target_sample, false, scale);
				} else if(rgb_tensor.maps() == 1) {
					// Tensor is Grayscale
					bool success = Tensor::CopyMap(rgb_tensor, source_sample, 0, target, target_sample, 0, false, scale);
					success &= Tensor::CopyMap(rgb_tensor, source_sample, 0, target, target_sample, 1, false, scale);
					success &= Tensor::CopyMap(rgb_tensor, source_sample, 0, target, target_sample, 2, false, scale);
          return success;
				} else {
					FATAL("Tensors with map count other than 1 or 3 are not supported!");
				}
			}
		} else {
      LOGDEBUG << "Sample " << source_index << " requested";
			return false;
		}
  }
  
	unsigned int ListTensorStream::LoadFiles(std::string image_list_fname, std::string image_directory, std::string label_list_fname, std::string label_directory) {
		unsigned int number_of_classes = class_manager_->GetMaxClassId() + 1;
		bool dont_load_labels = false;

		if(label_list_fname.compare("DONOTLOAD") == 0) {
			label_list_fname = image_list_fname;
			label_directory = image_directory;
			dont_load_labels = true;
		}
		
		if(image_directory.back() != '/')
			image_directory += "/";

		if(label_directory.back() != '/')
      label_directory += "/";

		// Open file lists
		std::ifstream image_list_file ( image_list_fname, std::ios::in );

		if ( !image_list_file.good() ) {
			FATAL ( "Cannot open image list file: " << image_list_fname );
		}

		std::ifstream label_list_file ( label_list_fname, std::ios::in );

		if ( !label_list_file.good() ) {
			FATAL ( "Cannot open label list file: " << label_list_fname );
		}
		
		unsigned int tensor_count = 0;
		
		// Iterate through lists of images and labels
		while ( !image_list_file.eof() ) {
			std::string image_fname;
			std::string label_fname;
			std::getline ( image_list_file, image_fname );
			std::getline ( label_list_file, label_fname );

			if ( image_fname.length() < 5 || label_fname.length() < 5 )
				break;

			// LOGDEBUG << "Importing files " << image_fname << " and " << label_fname << "...";
			Conv::Tensor image_tensor ( image_directory + image_fname );
			Conv::Tensor label_rgb_tensor;

			if(!dont_load_labels) {
				label_rgb_tensor.LoadFromFile(label_directory + label_fname);

				if (image_tensor.width() != label_rgb_tensor.width() ||
						image_tensor.height() != label_rgb_tensor.height()) {
					LOGERROR << "Dimensions don't match, skipping file!";
					continue;
				}
			}

      if(image_tensor.maps() != 1 && image_tensor.maps() != 3) {
				FATAL("Map counts other than 1 or 3 are not supported! File: " << image_fname);
			}
			ListTensorMetadata image_md(image_directory + image_fname, image_tensor.width(), image_tensor.height(), 3, image_tensor.samples());
			ListTensorMetadata label_md(label_directory + label_fname, label_rgb_tensor.width(), label_rgb_tensor.height(), number_of_classes, label_rgb_tensor.samples());

      if(dont_load_labels)
				label_md.ignore = true;

			tensors_.push_back(image_md);
			tensors_.push_back(label_md);
			
			tensor_count += 2;
		}
		
		return tensor_count;
	}
}
