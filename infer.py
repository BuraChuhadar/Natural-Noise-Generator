import os
import numpy as np
import tensorflow as tf


def infer(args):
    from model import WaveGANGenerator

    infer_dir = os.path.join(args.train_dir, 'infer')
    if not os.path.isdir(infer_dir):
        os.makedirs(infer_dir)
    # Input zo
    z = tf.compat.v1.placeholder(tf.float32, [None, args.wavegan_latent_dim], name='z')
    y = tf.compat.v1.placeholder(tf.float32, [None, args.wavegan_smooth_len,1], name='y')

    # Execute generator
    with tf.compat.v1.variable_scope('G'):
        G_z = WaveGANGenerator(y, z,
                               args.wavegan_kernel_len,
                               args.wavegan_smooth_len,
                               args.wavegan_dim,
                               args.wavegan_batchnorm,
                               train=False)
    G_z = tf.identity(G_z, name='G_z')

    # Create saver
    G_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='G')
    global_step = tf.compat.v1.train.get_or_create_global_step()
    saver = tf.compat.v1.train.Saver(G_vars + [global_step])

    # Export graph
    tf.io.write_graph(tf.compat.v1.get_default_graph(), infer_dir, 'infer.pbtxt')





    # Export MetaGraph
    infer_metagraph_fp = os.path.join(infer_dir, 'infer.meta')
    tf.compat.v1.train.export_meta_graph(filename=infer_metagraph_fp,clear_devices=True,saver_def=saver.as_saver_def())

    # Reset graph (in case training afterwards)
    tf.compat.v1.reset_default_graph()


if __name__ == '__main__':
    import argparse
    from main import argument

    parser = argparse.ArgumentParser(description='script for infering')
    arg = argument(parser)
    args = arg.args_init()

    infer(args)
