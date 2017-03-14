import caffe
import numpy as np
import sys

if len(sys.argv) != 4:
  print 'USAGE:', sys.argv[0], '<binaryproto file> <CNParam file> <layer name>'
  quit()

binaryproto_file = sys.argv[1]
cnparam_file = sys.argv[2]
layer_name = sys.argv[3]

# Open mean image
blobproto = caffe.proto.caffe_pb2.BlobProto()
with open(binaryproto_file, 'rb') as f:
  blobproto.ParseFromString(f.read())

np_mean_image = caffe.io.blobproto_to_array(blobproto)

# Write CNParam header
cnparam = open(cnparam_file, 'wb')
np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(0xC240C240C240C240)).tofile(cnparam)

# Write layer name length
np.ndarray(shape=(1),dtype=np.uint32, buffer=np.array(len(layer_name))).tofile(cnparam)
# Write parameter set size
np.ndarray(shape=(1),dtype=np.uint32, buffer=np.array(1)).tofile(cnparam)
# Write name
cnparam.write(layer_name)

# Write weight (shape): samples, width, height, maps
np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(blobproto.num)).tofile(cnparam)
np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(blobproto.width)).tofile(cnparam)
np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(blobproto.height)).tofile(cnparam)
np.ndarray(shape=(1),dtype=np.uint64, buffer=np.array(blobproto.channels)).tofile(cnparam)

# Write weight (buffer)
#np.ndarray(shape=np_mean_image.shape, dtype=np.float32, buffer=np_mean_image).tofile(cnparam)
np_mean_image.astype(np.float32).tofile(cnparam)
cnparam.close()
