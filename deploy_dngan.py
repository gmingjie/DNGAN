"""
Deployment code for DBT denoising. 

2021.5.7 Mingjie Gao (gmingjie@umich.edu)
"""


from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import time
from models import Denoiser_DNGAN
from utils_functions import binread, binwrite, emptyStruct


def deploy(args):
    img = binread(args.input_file, chat=True)

    # Two magic numbers to "center" the histograms of the image
    image_mean = 0.049
    image_std = 0.0032
    img = (img - image_mean) / image_std

    input_w, input_h, nz = img.shape
    X = tf.placeholder(dtype=tf.float32, shape=[1, input_w, input_h, 1])
    with tf.variable_scope('generator_model', reuse=tf.AUTO_REUSE) as scope:
        Y = Denoiser_DNGAN(X)

    gen_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator_model')
    saver_gen = tf.train.Saver(var_list=gen_params)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    img_denoised = np.zeros((input_w, input_h, nz), dtype=np.float32)
    with tf.Session(config=config) as sess:
        print("--> Checkpoint path: %s" % args.weight_path)
        saver_gen.restore(sess, args.weight_path) 
        
        start_time = time.time()
        print('Denoising slices... ')
        for iz in range(nz):
            img_iz = img[:,:,iz:(iz+1)]
            img_iz = np.transpose(img_iz, (2, 0, 1))
            inputdata = np.expand_dims(img_iz, axis=3)
            outputdata = sess.run(Y, feed_dict={X:inputdata})

            imgout = np.squeeze(outputdata, axis=3)
            imgout = np.transpose(imgout, (1, 2, 0))
            img_denoised[:,:,iz:(iz+1)] = imgout

        img_denoised = img_denoised * image_std + image_mean
        elapsed_time = time.time() - start_time
        print('Time took: ' + str(elapsed_time))

    # Output recon volumes 
    binwrite(args.output_path, img_denoised, chat=True)

        
if __name__ == '__main__':
    args = emptyStruct()

    # Path to the input image (support .rawg format)
    args.input_file = './VICTRE_phantom/exampleVictrePhantomSlices_sart3_640x975x5.rawg'
    # Path to the trained model
    args.weight_path = './trained_model/24mAsToNoiseless'
    # Output file name (support .rawg format)
    args.output_path = './VICTRE_phantom/exampleVictrePhantomSlices_sart3_24ToInfdenoised_640x975x5.rawg'

    deploy(args)
