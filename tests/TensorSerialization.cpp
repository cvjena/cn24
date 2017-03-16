/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#include <cn24.h>
#include <random>

int main() {
  const unsigned int elements = 100;
  Conv::System::Init();
  
  // (1) Sanity check
  Conv::Tensor a(2, elements / 2), b(2, elements / 2), c(2, elements / 2);
  a.Clear(0); b.Clear(0); c.Clear(1);
  
  Conv::AssertEqual(a, b, "Sanity check");
  Conv::AssertNotEqual(a, c, "Sanity check");
  
  // (2) Fill A with random numbers
  std::mt19937_64 gen(12345);
  std::uniform_real_distribution<Conv::datum> dist(-1,1);
  
  for(unsigned int e = 0; e < elements; e++) {
    a(e) = dist(gen);
  }
  
  // (3) Test binary serialization
  std::ofstream tmp_out("test_TensorSerialization_tmp", std::ios::out | std::ios::binary);
  a.Serialize(tmp_out);
  tmp_out.close();
  
  std::ifstream tmp_in("test_TensorSerialization_tmp", std::ios::in | std::ios::binary);
  b.Deserialize(tmp_in);
  
  Conv::AssertEqual(a, b, "Binary serialization");
  
  // Reset b
  b.Clear(0);
  
  // (4) Test base64 serialization
  std::string tmp_base64 = a.ToBase64();
  bool result = b.FromBase64(tmp_base64);
  Conv::AssertEqual(true, result, "Deserialization return value");
  Conv::AssertEqual(a, b, "Base64 serialization");
  
  // Reset b
  b.Clear(0);
  
  // (5) Test sample-wise base64 serialization
  std::string tmp_s1 = a.ToBase64(0);
  std::string tmp_s2 = a.ToBase64(1);
  result = b.FromBase64(tmp_s1, 0);
  result &= b.FromBase64(tmp_s2, 1);
  Conv::AssertEqual(true, result, "Deserialization return value (sample-wise)");
  Conv::AssertEqual(a, b, "Base64 serialization (sample-wise)");
  
  LOGEND;
  return 0;
}
