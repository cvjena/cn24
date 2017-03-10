import sys
import numpy as np
import caffe
import inspect
import json
import google.protobuf as pb
from caffe.proto import caffe_pb2

# Parse command line arguments
if len(sys.argv) != 4:
  print 'USAGE:', sys.argv[0], '<prototxt> <caffemodel> <output name>'
  quit()

output_name = sys.argv[3]
prototxt_name = sys.argv[1]
caffemodel_name = sys.argv[2]

# Load Caffe model and prototxt
net_p = caffe_pb2.NetParameter()
with open(prototxt_name,'rb') as f:
  pb.text_format.Merge(str(f.read()), net_p)

# Instantiate Caffe model
net = caffe.Net(prototxt_name, caffemodel_name, caffe.TEST)

# Write CNParam header
cnparam = open(output_name + '.CNParam', 'wb')
np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(0xC240C240C240C240)).tofile(cnparam)

# Prepare JSON structure
outer_json = {}
input_json = {}
net_json = {}
nodes_json = {}
hyper_json = {}

# Iterate layers
last_layer_name = ""
input_layer_name = ""
for layer_n in net_p.layer:
  name = layer_n.name
  layer = net.layer_dict[name]
  layer_json = {}
  if(layer.type == "Convolution"):
    # Write layer name length
    np.ndarray(shape=(1),dtype=np.uint32, buffer=np.array(len(name))).tofile(cnparam)
    # Write parameter set size
    np.ndarray(shape=(1),dtype=np.uint32, buffer=np.array(2)).tofile(cnparam)
    # Write name
    cnparam.write(name)

    # Write weight (shape): samples, width, height, maps
    weight_blob = net.params[name][0]
    np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(weight_blob.num)).tofile(cnparam)
    np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(weight_blob.width)).tofile(cnparam)
    np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(weight_blob.height)).tofile(cnparam)
    np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(weight_blob.channels)).tofile(cnparam)
    # Write weight (buffer)
    weight_blob.data.tofile(cnparam)
    
    # Write bias (shape): samples, width, height, maps
    bias_blob = net.params[name][1]
    np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(bias_blob.num)).tofile(cnparam)
    np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(bias_blob.width)).tofile(cnparam)
    np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(bias_blob.height)).tofile(cnparam)
    np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(bias_blob.channels)).tofile(cnparam)
    # Write bias (buffer)
    bias_blob.data.tofile(cnparam)
    

    # Write layer params to JSON
    layer_json["type"] = "convolution"
    layer_json["group"] = layer_n.convolution_param.group
    layer_json["kernels"] = layer_n.convolution_param.num_output
    layer_json["size"] = [layer_n.convolution_param.kernel_size[0],layer_n.convolution_param.kernel_size[0]]
    if len(layer_n.convolution_param.stride) > 0:
      layer_json["stride"] = [layer_n.convolution_param.stride[0],layer_n.convolution_param.stride[0]]
    if len(layer_n.convolution_param.pad) > 0:
      layer_json["pad"] = [layer_n.convolution_param.pad[0],layer_n.convolution_param.pad[0]]

  elif(layer.type == "Pooling"):
    layer_json["type"] = "advanced_maxpooling"
    layer_json["size"] = [layer_n.pooling_param.kernel_size,layer_n.pooling_param.kernel_size]
    layer_json["stride"] = [layer_n.pooling_param.stride,layer_n.pooling_param.stride]
    if layer_n.pooling_param.pool != 0:
      print 'Unknown pooling type:', layer_n.pooling_param.pool

  elif(layer.type == "InnerProduct"):
    # Write layer name length
    np.ndarray(shape=(1),dtype=np.uint32, buffer=np.array(len(name))).tofile(cnparam)
    # Write parameter set size
    np.ndarray(shape=(1),dtype=np.uint32, buffer=np.array(2)).tofile(cnparam)
    # Write name
    cnparam.write(name)

    # Write weight (shape): samples, width, height, maps
    weight_blob = net.params[name][0]
    np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(weight_blob.num)).tofile(cnparam)
    np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(weight_blob.width)).tofile(cnparam)
    np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(weight_blob.height)).tofile(cnparam)
    np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(weight_blob.channels)).tofile(cnparam)
    # Write weight (buffer)
    weight_blob.data.tofile(cnparam)
    
    # Write bias (shape): samples, width, height, maps
    bias_blob = net.params[name][1]
    np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(bias_blob.num)).tofile(cnparam)
    np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(bias_blob.width)).tofile(cnparam)
    np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(bias_blob.height)).tofile(cnparam)
    np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(bias_blob.channels)).tofile(cnparam)
    # Write bias (buffer)
    bias_blob.data.tofile(cnparam)
    
    layer_json["type"] = "convolution"
    layer_json["kernels"] = layer_n.inner_product_param.num_output
    layer_json["size"] = [1,1]

  elif(layer.type == "Dropout"):
    layer_json["type"] = "dropout"
    layer_json["dropout_fraction"] = layer_n.dropout_param.dropout_ratio
  elif(layer.type == "LRN"):
    layer_json["type"] = "local_response_normalization"
    layer_json["alpha"] = layer_n.lrn_param.alpha
    layer_json["beta"] = layer_n.lrn_param.beta
    layer_json["size"] = layer_n.lrn_param.local_size
  elif(layer.type == "ReLU"):
    layer_json = "relu"
  elif(layer.type == "Softmax"):
    layer_json = "softmax"
  elif(layer.type == "Input"):
    input_json["width"] = layer_n.input_param.shape[0].dim[2]
    input_json["height"] = layer_n.input_param.shape[0].dim[3]
    last_layer_name = name
    input_layer_name = name
    continue
  else:
    print 'Unknown layer:', layer.type
    continue

  node_json = {}
  node_json["layer"] = layer_json
  if len(last_layer_name) > 0:
    if last_layer_name == input_layer_name:
      net_json["input"] = name
    else:
      node_json["input"] = last_layer_name
  last_layer_name = name
  nodes_json[name] = node_json

# Piece together JSON structure
net_json["output"] = last_layer_name
net_json["nodes"] = nodes_json
net_json["error_layer"] = "dummy"
outer_json["net"] = net_json
outer_json["input"] = input_json
outer_json["hyperparameters"] = hyper_json

# Write JSON to file
with open(output_name + '.json', 'w') as fj:
  fj.write(json.dumps(outer_json, indent=2))

cnparam.close()

print 'Done.'
