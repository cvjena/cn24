/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#ifndef CONV_COMPRESSEDTENSOR_H
#define CONV_COMPRESSEDTENSOR_H

#include <cstddef>
#include <string>
#include <iostream>

#include "Log.h"
#include "Config.h"

#include "Tensor.h"

namespace Conv {

class CompressedTensor;
/**
 * @brief Prints size to the ostream, may be helpful.
 */
std::ostream& operator<< (std::ostream& output, const CompressedTensor& tensor);

class CompressedTensor {
public:
  /**
   * @brief Constructs an empty CompressedTensor of zero size.
   */
  CompressedTensor ();
  
  ~CompressedTensor ();
  
  /*
   * Compression and decompression encapsulated
   */
  void Compress(Tensor& tensor);
  void Decompress(Tensor& tensor, datum* preallocated_memory = nullptr);


  /**
   * @brief Serializes the CompressedTensor to the stream.
   *
   * @param output The output stream
   * @param convert Convert to byte
   */
  void Serialize (std::ostream& output);

  /**
   * @brief Deserializes from the stream.
   *
   * Note that this resizes the stream if necessary and overwrites its content.
   * @param input The input stream
   * @param head_only Set to true to only read the dimensions
   * @param try_mmap Set to true to attempt to memory map the file
   * @param fd File descriptor for the SAME file as input's underlying
   */
  void Deserialize (std::istream& input, bool head_only = false, bool try_mmap = false, int fd = 0);
  
	/**
	 * @brief Writes some tensor statistics to the debug output
	 */
	void PrintStats();

  /**
   * @brief Deallocates the memory if data_ptr is not a nullptr.
   */
  void DeleteIfPossible();

  // Accessors for the size information
  inline std::size_t samples() const {
    return samples_;
  }
  inline std::size_t maps() const {
    return maps_;
  }
  inline std::size_t height() const {
    return height_;
  }
  inline std::size_t width() const {
    return width_;
  }
  inline std::size_t elements() const {
    return elements_;
  }
  inline std::size_t compressed_length() const {
    return compressed_length_;
  }

private:
  /**
   * @brief Resizes the CompressedTensor with data loss.
   */
  void Resize (const std::size_t samples, const std::size_t width,
               const std::size_t height, const std::size_t maps,
               const std::size_t compressed_length,
               char* const preallocated_memory = nullptr, bool mmapped = false );

  // Pointer to the actual data
  char* compressed_data_ptr_ = nullptr;

  // Sizes
  std::size_t samples_ = 0;
  std::size_t maps_ = 0;
  std::size_t height_ = 0;
  std::size_t width_ = 0;
  std::size_t elements_ = 0;
  
  std::size_t compressed_length_ = 0;
  
  static void CompressData(void* uncompressed, const std::size_t& uncompressed_elements, void* compressed, std::size_t& compressed_length);
  static void DecompressData(void* uncompressed, std::size_t& uncompressed_elements, void* compressed, const std::size_t& compressed_length);
  
public:
  
  /**
   * @brief If this is true, the CompressedTensor was memory mapped
   */
  bool mmapped_ = false;
  void* original_mmap_ = nullptr;
};



}

#endif
