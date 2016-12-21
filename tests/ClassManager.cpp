/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cn24.h>

int main() {
  Conv::System::Init();

  std::vector<std::string> sample_classes = {
      "Test", "Test1", "Test2", "Test3", "Test4"
  };
  Conv::ClassManager class_manager;
  Conv::ClassManager class_manager2;

  // Check if empty class manager is actually empty
  Conv::AssertEqual<int>(0, class_manager.GetMaxClassId(), "GetMaxClassId() on empty ClassManager");
  Conv::AssertEqual<int>(0, class_manager.GetClassCount(), "GetClassCount() on empty ClassManager");

  // Register classes
  for(unsigned int c = 0; c < sample_classes.size(); c++) {
    Conv::AssertEqual<bool>(true, class_manager.RegisterClassByName(sample_classes[c], 1, 2.0), "result of RegisterClassByName");
  }

  // See if classes come up as known, save max class id while at it
  unsigned int max_class_id = 0;
  for(unsigned int c = 0; c < sample_classes.size(); c++) {
    unsigned int class_id = class_manager.GetClassIdByName(sample_classes[c]);
    Conv::AssertNotEqual<int>(UNKNOWN_CLASS, class_id, "result of GetClassIdByName");
    if(class_id > max_class_id)
      max_class_id = class_id;

    // See if class info that comes up by id is correct
    std::string resolved_class_name = class_manager.GetClassInfoById(class_id).first;
    if(resolved_class_name.compare(sample_classes[c]) != 0) {
      FATAL("Assertion failed: Expected GetClassInfoById.name to be " << sample_classes[c] << ", actual value: " << resolved_class_name);
    }
  }

  Conv::AssertEqual<int>(max_class_id, class_manager.GetMaxClassId(), "result of GetMaxClassId");

  // See if dumping and loading again works
  class_manager2.LoadFromFile(class_manager.SaveToFile());
  for(unsigned int c = 0; c < sample_classes.size(); c++) {
    Conv::AssertEqual<int>(class_manager.GetClassIdByName(sample_classes[c]), class_manager2.GetClassIdByName(sample_classes[c]), "result of dumped and reloaded GetClassIdByName");
    Conv::AssertEqual<int>(class_manager.GetClassInfoByName(sample_classes[c]).color, class_manager2.GetClassInfoByName(sample_classes[c]).color, "result of dump and loaded GetClassInfoByName.color");
    Conv::AssertEqual<Conv::datum>(class_manager.GetClassInfoByName(sample_classes[c]).weight, class_manager2.GetClassInfoByName(sample_classes[c]).weight, "result of dump and loaded GetClassInfoByName.weight");
  }

  LOGEND;
  return 0;
}

