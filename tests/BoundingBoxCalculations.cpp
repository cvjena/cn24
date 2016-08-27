/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#include <cn24.h>

int main() {
  Conv::System::Init();

  // Sample Boxes
  Conv::BoundingBox A(2.5, 6.25, 1, 2.5);
  Conv::BoundingBox B(4.75, 2, 5.5, 2);
  Conv::BoundingBox C(7.5, 2.5, 3, 4);
  Conv::BoundingBox D(7.75, 6.75, 2.5, 2.5);
  Conv::BoundingBox E(7.5, 6.5, 1, 1);

  // Boxes B and C
  Conv::datum BC_HorOver = Conv::BoundingBox::Overlap1D(B.x, B.w, C.x, C.w);
  Conv::datum BC_VertOver = Conv::BoundingBox::Overlap1D(B.y, B.h, C.y, C.h);
  Conv::datum BC_Intersection = B.Intersection(&C);
  Conv::datum BC_Union = B.Union(&C);

  Conv::AssertEqual((Conv::datum)1.5, BC_HorOver, "BC_HorOver");
  Conv::AssertEqual((Conv::datum)2, BC_VertOver, "BC_VertOver");
  Conv::AssertEqual((Conv::datum)3, BC_Intersection, "BC_Intersection");
  Conv::AssertEqual((Conv::datum)20, BC_Union, "BC_Union");

  // Boxes A and D
  Conv::datum AD_HorOver = Conv::BoundingBox::Overlap1D(A.x, A.w, D.x, D.w);
  Conv::datum AD_VertOver = Conv::BoundingBox::Overlap1D(A.y, A.h, D.y, D.h);
  Conv::datum AD_Intersection = A.Intersection(&D);
  Conv::datum AD_Union = A.Union(&D);

  Conv::AssertLess((Conv::datum)0, AD_HorOver, "AD_HorOver");
  Conv::AssertEqual((Conv::datum)2, AD_VertOver, "AD_VertOver");
  Conv::AssertEqual((Conv::datum)0, AD_Intersection, "AD_Intersection");
  Conv::AssertEqual((Conv::datum)8.75, AD_Union, "AD_Union");

  // Boxes D and E
  Conv::datum DE_HorOver = Conv::BoundingBox::Overlap1D(D.x, D.w, E.x, E.w);
  Conv::datum DE_VertOver = Conv::BoundingBox::Overlap1D(D.y, D.h, E.y, E.h);
  Conv::datum DE_Intersection = D.Intersection(&E);
  Conv::datum DE_Union = D.Union(&E);

  Conv::AssertEqual((Conv::datum)1, DE_HorOver, "DE_HorOver");
  Conv::AssertEqual((Conv::datum)1, DE_VertOver, "DE_VertOver");
  Conv::AssertEqual((Conv::datum)1, DE_Intersection, "DE_Intersection");
  Conv::AssertEqual((Conv::datum)6.25, DE_Union, "DE_Union");


  LOGEND;
  return 0;
}
