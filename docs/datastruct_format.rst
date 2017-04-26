Data Format
...........
Data is provided to CN24 in the form of serialized Bundles.
The serialization method of choice is JSON, provided by `nlohnmann's JSON library <https://github.com/nlohmann/json>`_.

The Bundle format is best explained by an example:

.. code-block:: json

    {
      "name": "SampleBundle",
      "segments:" [
        {
          "name": "SampleSegmentA",
          "samples": [ ]
        },
        {
          "name": "SampleSegmentB",
          "samples": [ ]
        }
      ]
    }

The samples themselves are JSON objects as well. Their exact
schema depends on the task.

Detection
~~~~~~~~~
CN24 supports detection using the `YOLO method <https://arxiv.org/abs/1506.02640>`_.
Samples need to specify the following:

* **image_filename**: Input image file
* **boxes**: JSON array of bounding boxes

Bounding boxes have the following properties:

* **x**, **y**: Coordinates of the *center* of the bounding box (pixels)
* **w**, **h**: Width and height of the bounding box (pixels)
* **class**: Class of the object inside the bounding box

Optionally, you can specificy these:

* **difficult**: If set to 1, the box is ignored during testing
* **dont_scale**: Instead of pixels, the coordinates and dimensions
  of the box are specified as normalized fractions of the image dimensions

The following is an
example from the PASCAL VOC dataset:

.. code-block:: json

  {
    "boxes": [
      {
        "class": "bird",
        "difficult": 0,
        "h": 286,
        "w": 156,
        "x": 338,
        "y": 190
      }
    ],
    "image_filename": "2011_003213.jpg"
  }

Classification
~~~~~~~~~~~~~~

Binary Segmentation
~~~~~~~~~~~~~~~~~~~
Samples for binary segmentation consist of two image files
with equal dimensions. One is the actual input image and the
other the label image. At the moment, only binary segmentation
is supported. Grayscale label images are preferred. However,
CN24 will also accept RGB images as labels. In this case, the
value of the third channel will be used as a label.

The following properties need to be specified:

* **image_filename**: Input image file
* **label_filename**: Label file

Optionally, you can supply a value for **localized_error_function**.
Currently, the only supported values are *default* and *kitti*.

The following is an example from the KITTI-Vision Road Dataset:

.. code-block:: json

  {
    "label_filename": "gt_image_2/umm_road_000049.png",
    "localized_error_function": "kitti", 
    "image_filename": "image_2/umm_000049.png"
  }
