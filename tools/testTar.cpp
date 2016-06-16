/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */


#include <cn24.h>

int main(int argc, char* argv[]) {
  Conv::System::Init();
  if(argc!=2) {
    LOGINFO << "USAGE: " << argv[0] << " <tar file>";
    LOGEND;
    return 0;
  }

  Conv::MemoryMappedFile* mmfile = new Conv::MemoryMappedFile(std::string(argv[1]));
  Conv::MemoryMappedTar* mmtar = new Conv::MemoryMappedTar(mmfile->GetAddress(), mmfile->GetLength());

  unsigned long mmfile_begin = (unsigned long)mmfile->GetAddress();
  unsigned long mmfile_end = mmfile_begin + (unsigned long)mmfile->GetLength();

  for(unsigned int f = 0; f < mmtar->GetFileCount(); f++) {
    const Conv::MemoryMappedTarFileInfo& info = mmtar->GetFileInfo(f);
    LOGINFO << "File: " << info.filename << ", length: " << info.length << ", address: " << info.data;
    if(((unsigned long)info.data) < mmfile_begin || ((unsigned long)info.data) >= mmfile_end) {
      LOGERROR << "Out of bounds!";
    }
  }

  delete mmfile;
  LOGEND;
  return 0;
}