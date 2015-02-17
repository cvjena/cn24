This manual is a collection of the most important wiki articles. Please visit
the wiki for more up-to-date articles.

# Building CN24

## Dependencies
CN24 uses the CMake cross-platform build system. You need at least version
2.8 to generate the build files.
The following compilers are supported for building CN24:
* GCC >= 4.8
* Clang >= 3.5
* Visual Studio >= 2013

Older versions will probably work as long as they support the C++11 features
used by CN24. All other dependencies are optional. Optional dependencies include:
* _libjpeg_ and _libpng_ to read .jpg and .png files
* _Intel MKL_, _AMD ACML_ or _ATLAS_ for faster calculations
* _OpenCL_ for GPU acceleration
* _GTK+ 3_ for GUI utilities

## Building CN24
First, make sure you have all the required dependencies. Then, clone the
CN24 repository:

```bash
git clone https://github.com/cvjena/cn24.git
```

Create a build directory and run CMake:

```bash
mkdir build
cmake path/to/cn24
```

Run your preferred build tool, for example:
```
make
```

That's it, you're done!

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

# Specifying an Architecture
For maximum flexibility without creating further dependencies, CN24 uses its own language to specify neural network architectures. 

To design a new network architecture, create an empty file (preferred extension: _.net_).

## Adding Layers
Layers are specified one at a time, in the order in which they are evaluated during a forward pass.
There are several types of layers supported by CN24, each with their own set of parameters.
You can add your own layers to the language by modifying the [code](https://github.com/cvjena/cn24/blob/master/src/factory/ConfigurableFactory.cpp#L75).

In general, a layer is specified by a single line beginning with a question mark, followed by the type of
layer and an optional list of parameters:
```
?layer_type param1=a param2=b
```
### Convolutional Layers
In fully convolutional network, these layers are the only layers that have weights.
These connection weights are many three-dimensional convolution kernels used in a "valid" type convolution.
Convolutional layers are specified using the following command:
```
?convolutional size=5x5 kernels=8
```
The _size_ parameter describes the dimensions of the individual convolution kernels.
The third dimension is setup automatically because it needs to match the previous layer.
_kernels_ specifies the number of individual three-dimensional convolution kernels in the layer.
This is the number of feature maps coming out of the layer.

_Note:_ Using odd numbers in the _size_ parameter is preferred because it leads to receptive fields with
even dimensions which are easier to process. However, this is not enforced by CN24.

There is a special command for convolutional layers with 1x1 kernels:
```
?fullyconnected neurons=100
```
In a network processing individual patches, a 1x1 convolutional layer would be equal to a fully connected
layer. The _neurons_ parameter corresponds to the _kernels_ parameter of the convolutional layer.

### Maximum Pooling Layers
Spatial pooling layers divide their input into equally sized regions.
Each output pixel represents the region through its value. Maximum pooling uses the maximum to represent a
region. The pooling is applied to each feature map separately. Maximum pooling layers are specified
using the following command:
```
?maxpooling size=2x2
```

### Nonlinearities
CN24 supports the most common nonlinear activation functions. Use one of the following commands to add a
nonlinearity layer:
```
?relu
?sigm
?tanh
```

### Special layers
#### Spatial prior
During the research for our paper [Convolutional Patch Networks with Spatial Prior for Road Detection and
Urban Scene Understanding](http://hera.inf-cv.uni-jena.de:6680/pdf/Brust15:CPN.pdf) we found that adding
spatial context in form of coordinates improves segmentation performance for certain tasks.
The spatial prior layer adds two feature maps, one for the horizontal and one for the vertical pixel coordinates. Use the following command to add a spatial prior layer:
```
?spatialprior
```

## Hyperparameters
The default hyperparameters used during training can be overridden in the configuration file. This is
recommended, because optimal hyperparameters are highly dependent on the architecture and training data.
The following is an example section containing the default values:
```
l1=0.001
l2=0.0005
lr=0.0001
gamma=0.003
momentum=0.9
exponent=0.75
iterations=500
```

## Example
This is one of the configurations used for road detection on the KITTI dataset. You can use this
example as a starting point for your own configurations:
```
# Sample CNN for KITTI Dataset

# Network configuration
?convolutional kernels=12 size=7x7
?maxpooling size=2x2
?relu

?convolutional size=5x5 kernels=6
?relu

?convolutional size=5x5 kernels=48
?relu

?fullyconnected neurons=192
?relu

?fullyconnected neurons=(o)
?output

# Learning settings
l1=0.001
l2=0.0005
lr=0.0001
gamma=0.003
momentum=0.9
exponent=0.75
iterations=500
``` 
