/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#ifndef CONV_JSONPARSING_H
#define CONV_JSONPARSING_H

#include "../../../external/json/json.hpp"

#define JSON_TRY_DATUM(target_var, json_object, key, otherwise) if(json_object.count(key) == 1 && json_object[key].is_number()) \
{ target_var = json_object[key]; } else { target_var = otherwise; }
#define JSON_TRY_INT(target_var, json_object, key, otherwise) if(json_object.count(key) == 1 && json_object[key].is_number_integer()) \
{ target_var = json_object[key]; } else { target_var = otherwise; }
#define JSON_TRY_BOOL(target_var, json_object, key, otherwise) if(json_object.count(key) == 1 && json_object[key].is_number_integer()) \
{ target_var = ((int)json_object[key] > 0); } else { target_var = otherwise; }
namespace Conv {
  using JSON = nlohmann::json;
}

#endif





















