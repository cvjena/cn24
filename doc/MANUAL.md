# Importing Datasets
To use your dataset with CN24, you need to prepare two things:

## 1. Dataset configuration
To import a dataset, CN24 needs to know how many classes your dataset
contains and what their names are. This information is supplied in a
dataset configuration file (ending: *.set*). Start with an empty file
and add the following:

```
classes=3
Vehicle
Sign
LaneMarking
```

Labels are supplied as images. If your dataset has polygon label data,
please rasterize it first. CN24 needs to know the classes' colors to
interpret the label images correctly. Add a section like this to your
dataset configuration file to specify the colors:

```
colors
0xFF0000
0x00FF00
0x0000FF
```

_Note:_ For two-class problems, you can also specify only one class
and color. The color of the negative class is assumed to be black.
This allows you to supply confidence maps as label data. CN24
will also display different statistics during training. In this case,
your dataset configuration should look like this:

```
classes=1
road
colors
0xFFFFFF
```

You need a dataset configuration file to import images.

## 2. Your images
Make sure you enable the CMake build options
*CN24_BUILD_PNG* or *CN24_BUILD_JPG* depending on your dataset's
file format. To achieve more predictable and constant performance,
CN24 reads images and labels from a Tensor Stream. A Tensor Stream is a
file that contains uncompressed floating-point image data. CN24 can
alternate between two Tensor Streams: one for training and one for
validation. 

First, collect your training images and labels. Create a file that
contains one image filename per line. The create another file that
contains one label filename per line, using matching line numbers
for corresponding image and label files.
From a shell, run the _makeTensorStream_ utility to convert your
data to the Tensor Stream format:
``` bash
makeTensorStream dataset.set image_filename images/ label_filenames labels/ DATASET_TRAIN.Tensor
```

This will create a file "DATASET\_TRAIN.Tensor" in the working directory.
Repeat this procedure for your validation data.

_Note:_ If you compile CN24 with the *CN24_BUILD_GUI* CMake build option,
you can use the _tensorTool_ utility to view the Tensor Streams. This
allows you verify the import.

## 3. Completing the dataset
CN24 needs to know where to find the Tensor Streams. Add the following
lines to your dataset configuration file to specify the locations:

```
training=~/dataset/DATASET_TRAIN.Tensor
testing~/dataset/DATASET_TEST.Tensor
```

Your dataset is now ready for use with CN24!
