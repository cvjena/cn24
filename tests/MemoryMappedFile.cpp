/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <fstream>

#include <cn24.h>

int main() {
  Conv::System::Init();

  const unsigned int test_length = 1024;
  const std::string test_filename = "tmp_test_memmappedfile";
  uint8_t* test_sequence = new uint8_t[test_length];

  for(unsigned int i = 0; i < test_length; i++) {
    test_sequence[i] = (uint8_t)(i % 100);
  }

  std::ofstream output_stream(test_filename, std::ios::out | std::ios::binary);
  if(!output_stream.good()) {
    LOGERROR << "Cannot write test data!";
    LOGEND;
    return -1;
  }

  output_stream.write((const char*)test_sequence, test_length * sizeof(uint8_t));
  output_stream.close();

  Conv::MemoryMappedFile* test_mmapped_file = new Conv::MemoryMappedFile(test_filename);
  uint8_t* mapped_test_sequence = (uint8_t*)test_mmapped_file->GetAddress();
  unsigned int mapped_test_length = test_mmapped_file->GetLength();

  if(mapped_test_sequence == nullptr) {
    LOGERROR << "File mapped at null address!";
    LOGEND;
    return -1;
  }

  if(mapped_test_length != test_length) {
    LOGERROR << "Wrong mapped file length, expected " << test_length << " but got " << mapped_test_length << "!";
    LOGEND;
    return -1;
  }

  for(unsigned int i = 0; i < test_length; i++) {
    if(mapped_test_sequence[i] != test_sequence[i]) {
      LOGERROR << "Error in test sequence at location " << i;
      LOGEND;
      return -1;
    }
  }


  delete test_mmapped_file;
  delete[] test_sequence;
  LOGEND;
  return 0;
}
