/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file Tensor.h
 * @class Tensor
 * @brief Represents vectors, matrices, 3-D and 4-D tensors.
 *
 * This class stores the data in the following hierarchy:
 *  1. Samples
 *  2. Channels
 *  3. Lines
 *  4. Pixels
 *
 * Note that this does not match the order in which the parameters
 * appear in the constructor and the Resize function.
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 *
 */

#ifndef CONV_TENSOR_H
#define CONV_TENSOR_H

#include <cstddef>
#include <string>
#include <iostream>

#include "Log.h"
#include "Config.h"

namespace Conv {

class Tensor;
/**
 * @brief Prints size to the ostream, may be helpful.
 */
std::ostream& operator<< (std::ostream& output, const Tensor& tensor);

class Tensor {
public:
  /**
   * @brief Constructs an empty Tensor of zero size.
   */
  Tensor ();

  /**
   * @brief Constructs a deep copy of the Tensor
   *
   * @param tensor Tensor to copy
   * @param intentional Set this to true if you really wanted to copy a Tensor
   */
  Tensor (const Tensor& tensor, bool intentional = false);

  /**
   * @brief Moves a Tensor.
   *
   * @param tensor Tensor to move
   */
  Tensor (Tensor && tensor);
  
  /**
   * @brief Loads the Tensor from a file
   * 
   * @param filename Full path of the file to load
   */
  explicit Tensor (const std::string& filename);
  
  /**
   * @brief Constructs an empty Tensor of the specified size.
   *
   * @param number Number of samples for batching
   * @param channels Number of channels/feature maps per sample
   * @param width Width of each feature map
   * @param height Height of each feature map
   */
  explicit Tensor (const std::size_t samples, const std::size_t width = 1,
                   const std::size_t height = 1, const std::size_t maps = 1);

  /**
   * Destructor
   */
  ~Tensor();

  /**
   * @brief Sets the whole Tensor to a specific value.
   *
   * @param value Value to set the Tensor to
   */
  void Clear (const datum value = 0, const int sample = -1);

  /**
   * @brief Uses the memory of another Tensor
   *
   * @param tensor Tensor to shadow
   */
  void Shadow (Tensor& tensor);

  /**
   * @brief Resizes the Tensor with data loss.
   */
  void Resize (const std::size_t samples, const std::size_t width = 1,
               const std::size_t height = 1, const std::size_t maps = 1,
               datum* const preallocated_memory = nullptr, bool mmapped = false,
               bool dont_delete = false);

  /**
   * @brief Extends the Tensor to hold more samples without data loss
   *
   * This is an expensive operation!
   * @param samples
   */
  void Extend (const std::size_t samples);

  /**
   * @brief Resizes the Tensor to match another Tensor's size.
   *
   * @param tensor The Tensor to take the size data from
   */
  void Resize (const Tensor& tensor);

  /**
   * @brief Resizes the Tensor without data loss.
   *
   * Note that this will fail if the number of elements doesn't match.
   */
  bool Reshape (const std::size_t samples, const std::size_t width = 1,
                const std::size_t height = 1, const std::size_t maps = 1);

  /**
   * @brief Gets the offset (element number) of a specific element.
   */
  inline std::size_t Offset (const std::size_t x, const std::size_t y,
                             const std::size_t map, const std::size_t sample)
  const {
    return (sample * maps_ * width_ * height_) +
           (map * width_ * height_) +
           (y * width_) +
           x;
  }


  /**
   * @brief Transposes every map in every sample.
   */
  void Transpose ();

  /**
   * @brief Serializes the Tensor to the stream.
   *
   * @param output The output stream
   * @param convert Convert to byte
   */
  void Serialize (std::ostream& output, bool convert = false);

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
   * @brief Loads a file and resizes the Tensor to match its contents
   * 
   * @param filename Full path of the file to load
   */
  void LoadFromFile(const std::string& filename);
  
  /**
   * @brief Writes the Tensor to a file
   * 
   * @param filename Full path of the file to write to
   */
  void WriteToFile(const std::string& filename);

	/**
	 * @brief Writes some tensor statistics to the debug output
	 */
	void PrintStats();

  /**
   * @brief Copies source to target Tensor (needs exact size match)
   * @param source Source Tensor
   * @param target Target Tensor
   * @return True on success
   */
  static bool Copy (const Tensor& source, Tensor& target);

  /**
   * @brief Copy a complete sample from one Tensor to another.
   *
   * @param source The source Tensor
   * @param source_sample The sample in the source Tensor
   * @param target The target Tensor
   * @param target_sample The sample in the target Tensor
   * @param allow_oversize If true, target size is allowed to be too small
   */
  static bool CopySample (const Tensor& source, const std::size_t source_sample,
                          Tensor& target, const std::size_t target_sample, const bool allow_oversize = false, const bool scale = false);

  /**
   * @brief Copy a map of a sample from one Tensor to another.
   *
   * @param source The source Tensor
   * @param source_sample The sample in the source Tensor
   * @param source_map The map in the source Tensor
   * @param target The target Tensor
   * @param target_sample The sample in the target Tensor
   * @param target_map The map in the target Tensor
   * @param allow_oversize If true, target size is allowed to be too small
   */
  static bool CopyMap (const Tensor& source, const std::size_t source_sample,
                       const std::size_t source_map, Tensor& target,
                       const std::size_t target_sample,
                       const std::size_t target_map, const bool allow_oversize = false, const bool scale = false);

  /**
   * @brief Return an interpolated value at a continuous location in the Tensor
   * @param x
   * @param y
   * @param map
   * @param sample
   *
   * @returns Interpolated value
   */
  datum GetSmoothData(datum x, datum y, std::size_t map, std::size_t sample) const;

  /**
   * @brief Deallocates the memory if data_ptr is not a nullptr.
   */
  void DeleteIfPossible();

  /**
   * @brief Gets the element number of the maximum in the specified sample.
   *
   * @param sample The sample to search
   */
  std::size_t Maximum (std::size_t sample);

  /**
   * @brief Gets the element number of the absolute maximum in the Tensor.
   */
  std::size_t AbsMaximum();
  
  /**
   * @brief Gets the map with the highest value at the specified position
   */
  std::size_t PixelMaximum(std::size_t x, std::size_t y, std::size_t sample);

  /**
   * @brief Get a const pointer to the data
   */
  inline const datum* data_ptr_const() const {
    return data_ptr_;
  }

  /**
   * @brief Get a pointer to the data
   */
  inline datum* data_ptr() const {
    return data_ptr_;
  }

  /**
   * @brief Get a pointer at a certain offset
   */
  inline datum* data_ptr (const std::size_t x, const std::size_t y = 0,
                          const std::size_t map = 0, const std::size_t sample = 0)
  const {
    return &data_ptr_[Offset (x, y, map, sample)];
  }

  /**
   * @brief Get a const pointer at a certain offset
   */
  inline const datum* data_ptr_const (const std::size_t x, const std::size_t y = 0,
                                      const std::size_t map = 0, const std::size_t sample = 0)
  const {
    return &data_ptr_[Offset (x, y, map, sample)];
  }

  /**
   * @brief Gets a reference to the specified element.
   */
  inline datum& operator[] (const std::size_t element) {
    return data_ptr_[element];
  }

  /**
   * @brief Gets a const reference to the specified element.
   */
  inline datum& operator() (const std::size_t element) const {
    return data_ptr_[element];
  }

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

private:
  // Pointer to the actual data
  datum* data_ptr_ = nullptr;
  bool is_shadow_ = false;
  Tensor* shadow_target_ = nullptr;

  // Sizes
  std::size_t samples_ = 0;
  std::size_t maps_ = 0;
  std::size_t height_ = 0;
  std::size_t width_ = 0;
  std::size_t elements_ = 0;
  
public:
  /**
   * @brief Moves the data to the CPU's memory if it isn't there already.
   * @param no_copy Don't copy the data
   */
  void MoveToCPU(bool no_copy = false);
  
  /**
   * @brief Moves the data to the GPU's memory if it isn't there already.
   * @param no_copy Don't copy the data
   */
  void MoveToGPU(bool no_copy = false);
  
  /**
   * @brief Pointer to the memory in the GPU's memory
   */
  void* cl_data_ptr_ = 0;
  
  /**
   * @brief If this is true, the data is currently in the GPU's memory
   */
  bool cl_gpu_ = false;
  
  /**
   * @brief If this is true, the Tensor was memory mapped
   */
  bool mmapped_ = false;
  void* original_mmap_ = nullptr;
  
  
  bool hint_ignore_content_ = false;
};



}

#endif
