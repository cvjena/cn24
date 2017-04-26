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
Samples need to specify an image file and bounding boxes. 
Bounding boxes have the following properties:

* **x**, **y**: Coordinates of the *center* of the bounding box
* **w**, **h**: Width and height of the bounding box
* **class**: Class of the object inside the bounding box

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
