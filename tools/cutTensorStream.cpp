/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <fstream>

#include <cn24.h>

int main(int argc, char* argv[]) {
  if (argc < 3) {
    LOGERROR << "USAGE: " << argv[0] << " <tensor stream file> <number of tensors to leave>";
    LOGEND;
    return -1;
  }

  Conv::System::Init();

  // Read tensor id from command line
  std::string s_tcnt(argv[2]);
  unsigned int t_count = atoi(s_tcnt.c_str());

  // Open tensor stream
  std::ifstream file_in(std::string(argv[1]), std::ios::in | std::ios::binary);

  // Seek until eof to count tensors
  Conv::Tensor tensor;
  unsigned int tensors_in_file = 0;
  for (; !file_in.eof(); tensors_in_file++) {
    tensor.Deserialize(file_in);
    file_in.peek();
    LOGINFO << "Tensor loaded: " << tensor;
  }

  // Compare the number of tensors in the stream to the specified output number
  if (tensors_in_file < t_count) {
    LOGERROR << "There are less than " << t_count << " Tensors in the specified file!";
    file_in.close();
  }
  else if (tensors_in_file == t_count) {
    LOGINFO << "Nothing to do here!";
    file_in.close();
  }
  else {
    // Rewind the stream
    file_in.clear();
    file_in.seekg(0, std::ios::beg);

    // Read in all tensors
    Conv::Tensor* tensors = new Conv::Tensor[t_count];
    for (unsigned int t = 0; t < t_count; t++)
      tensors[t].Deserialize(file_in);

    // Close the istream, open an ostream
    file_in.close();

    std::ofstream file_out(std::string(argv[1]), std::ios::out | std::ios::binary);
    for (unsigned int t = 0; t < t_count; t++)
    {
      tensors[t].Serialize(file_out);
      LOGINFO << "Serializing tensor " << t << ": " << tensors[t];
    }

    file_out.close();

    LOGINFO << "DONE!";
  }

  LOGEND;
  return 0;
}

