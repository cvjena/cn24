/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "BoundingBox.h"

namespace Conv {

datum BoundingBox::IntersectionOverUnion(BoundingBox *bounding_box) {
  return Intersection(bounding_box) / Union(bounding_box);
}

datum BoundingBox::Overlap1D(datum center1, datum size1, datum center2, datum size2) {
  datum left1 = center1 - size1 / (datum)2.0;
  datum left2 = center2 - size2 / (datum)2.0;
  datum inner_left = left1 > left2 ? left1 : left2;

  datum right1 = center1 + size1 / (datum)2.0;
  datum right2 = center2 + size2 / (datum)2.0;

  datum inner_right = right1 < right2 ? right1 : right2;

  return inner_right - inner_left;
}

datum BoundingBox::Intersection(BoundingBox *bounding_box) {
  datum horizontal_overlap = Overlap1D(x, w, bounding_box->x, bounding_box->w);
  datum vertical_overlap = Overlap1D(y, h, bounding_box->y, bounding_box->h);

  if(horizontal_overlap < 0 || vertical_overlap < 0) {
    return 0;
  } else {
    return horizontal_overlap * vertical_overlap;
  }
}

datum BoundingBox::Union(BoundingBox *bounding_box) {
  datum intersection = Intersection(bounding_box);
  return w * h + bounding_box->w * bounding_box->h - intersection;
}

bool BoundingBox::CompareScore(const BoundingBox &box1, const BoundingBox &box2) {
  return box1.score < box2.score;
}

}
