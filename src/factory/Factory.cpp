/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include "Factory.h"

namespace Conv {
  
Conv::Factory* Conv::Factory::getNetFactory (char net_id, const unsigned int seed) {
  switch(net_id) {
    case 'A':
      return new CNetAFactory(seed);
/*    case 'B':
      return new CNet8PlusFactory(seed);
    case 'C':
      return new CNet9PlusFactory(seed);
    case 'D':
      return new CNetDFactory(seed);
    case 'E':
      return new CNetEFactory(seed);
    case 'F':
      return new CNetFFactory(seed);
    case 'G':
      return new CNetGFactory(seed);
    case 'H':
      return new CNetHFactory(seed);
    case 'K':
      return new CNetKFactory(seed);
    case 'L':
      return new CNetLFactory(seed);*/
    case 'M':
      return new CNetMFactory(seed);
    case 'N':
      return new CNetNFactory(seed);
    case 'O':
      return new CNetOFactory(seed);
    case 'P':
      return new CNetPFactory(seed);
    case 'Q':
      return new CNetQFactory(seed);
    case 'R':
      return new CNetRFactory(seed);
    case 'S':
      return new CNetSFactory(seed);
/*    case '0':
      return new FastNetFactory(seed);*/
    default:
      return nullptr;
  }
}

}
