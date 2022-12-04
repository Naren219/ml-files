import gzip
import numpy as np

filename = 't10k-images-idx3-ubyte.gz'
num_images = 10
print('Extracting', filename)
with gzip.open(filename) as bytestream:
    bytestream.read
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
print(labels.shape)
