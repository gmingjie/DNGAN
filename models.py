"""
The DCNN structure. 

2021.5.7 Mingjie Gao (gmingjie@umich.edu)
"""


import tensorflow as tf


def Denoiser_DNGAN(inputs, nlayers=10, nchannels=32):
    def padConv(inputs, out_channels, kernel, name):
        padsize = (kernel - 1) / 2
        with tf.name_scope(name):
            outputs = tf.pad(inputs, [[0,0], [padsize,padsize], [padsize,padsize], [0,0]], 'REFLECT')
            outputs = tf.layers.conv2d(outputs, out_channels, kernel, padding='valid', kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), use_bias=True)
        return outputs

    outputs = inputs
    for ilayer in range(nlayers - 1):
        outputs = tf.nn.relu(padConv(outputs, nchannels, 3, name='conv{}'.format(ilayer + 1)))
    outputs = padConv(outputs, 1, 3, name='conv{}'.format(nlayers))

    outputs_final = inputs + outputs 
    return outputs_final