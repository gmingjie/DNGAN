"""
Some utility functions. 

2021.5.7 Mingjie Gao (gmingjie@umich.edu)
"""


import struct
import numpy as np


class emptyStruct:
    pass


def binread(imageName, chat=False):
    """
    Function to read .rawg (or .recon) files.
    """
    if chat: 
        print("Read from " + imageName)
    with open(imageName, "rb") as f:
        size = struct.unpack('<iii', f.read(12))
        fmt = "<" + str(1024 - 12) + "B"
        padding = struct.unpack(fmt, (f.read((1024 - 12) * 1)))
        img_size = size[0] * size[1] * size[2]
        fmt = "<" + str(img_size) + "f"
        img_tuple = struct.unpack(fmt, (f.read(img_size * 4)))
    img = np.array(img_tuple, dtype=np.float32)
    img = np.reshape(img, size, order='F')
    img = np.squeeze(img)
    return img


def binwrite(imageName, img, chat=False):
    """
    Function to write .rawg or .recon files.
    """ 
    if chat: 
        print("Write to " + imageName)
    sz = list(img.shape)
    if img.ndim == 2: 
        sz.append(1)
    with open(imageName, 'wb') as f:
        f.write(struct.pack('<iii', sz[0], sz[1], sz[2]))
        padding = np.ones(1024-12, dtype='<u1')
        f.write(padding.tobytes())
        f.write(img.astype('<f4').tobytes(order='F'))

