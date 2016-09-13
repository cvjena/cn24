/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_JSONDATASETFACTORY_H
#define CONV_JSONDATASETFACTORY_H

#include "../util/Dataset.h"
#include "../util/JSONParsing.h"

namespace Conv {

class JSONDatasetFactory {
public:
  static Dataset* ConstructDataset(JSON descriptor, ClassManager* class_manager);
};

}

#endif
