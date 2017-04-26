Hierarchy
.........

Data in CN24 is managed in a three-level hierachy:

(1) **Areas** designate the data's experimental purpose.
    There are 3 default areas: *training*, *staging* and *testing*.
(2) **Bundles** are the default unit of dataset serialization.
    They can be moved freely between areas. Bundles in the
    training Area can be assigned a weight that influences the
    likelihood of selecting training samples from them.
(3) **Segments** contain the samples themselves. They can be moved
    freely between Bundles. They exist to group samples, e.g.,
    training and validation samples or samples of different classes.

CN24 will create two empty default Bundles: *Default_Training* and
*Default_Testing*

::

        Area                        Bundle                       Segment   Samples
    Training
           |..............Default_Training                                      95
           |..................Weight:    1
                                         |.......................UM_road        95
  
     Staging
           |.............KITTIRoadTraining                                     193
                                         |.......................UM_lane        95
                                         |.......................UU_road        98
  
     Testing
           |...............Default_Testing                                      96
                                         |......................UMM_road        96


