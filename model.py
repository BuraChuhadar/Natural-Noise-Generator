import tensorflow as tf


def conv1d_transpose(inputs,filters,kernel_width,stride=4,padding='same'):
    inputs = tf.expand_dims(inputs, axis=1)
    result = tf.compat.v1.layers.conv2d_transpose(inputs,filters,(1, kernel_width),strides=(1, stride),padding='same')
    return result[:, 0]


"""
    Input: z:[None, 100],y:[None,4096]
    Output: [None, 32768, 1]
"""

def WaveGANGenerator(y,z,kernel_len=25,y_len=4096,dim=64,use_batchnorm=False,train=False):
    batch_size = tf.shape(z)[0]
    if use_batchnorm:
        batchnorm = lambda x: tf.compat.v1.layers.batch_normalization(x, training=train)
    else:
        batchnorm = lambda x: x

    # FC and reshape for convolution
    # [100] -> [16, 2048] in total 32768
    dim_mul = 32
    output = z
    with tf.compat.v1.variable_scope('z_project'):
        output = tf.compat.v1.layers.dense(output, 4 * 4 * dim * dim_mul)
        output = tf.reshape(output, [batch_size, 16, dim * dim_mul])
        output = batchnorm(output)
    output = tf.nn.relu(output)
    dim_mul //= 2 #dim_mul = 16

    # Layer 0
    # [16, 2048] -> [64, 1024]
    with tf.compat.v1.variable_scope('upconv_0'):
        output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4)
        output = batchnorm(output)
    output = tf.nn.relu(output)
    dim_mul //= 2 #dim_mul = 8

    # Layer 1
    # [64, 1024] -> [256, 512]
    with tf.compat.v1.variable_scope('upconv_1'):
        output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4)
        output = batchnorm(output)
    output = tf.nn.relu(output)
    dim_mul //= 2  #dim_mul = 4

    # Layer 2
    # [256, 512] -> [1024, 256]
    with tf.compat.v1.variable_scope('upconv_2'):
        output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4)
        output = batchnorm(output)
    output = tf.nn.relu(output)
    dim_mul //= 2 #dim_mul = 2

    # Layer 3
    # [1024, 256] -> [4096, 128]
    with tf.compat.v1.variable_scope('upconv_3'):
        output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4)
        output = batchnorm(output)
    output = tf.nn.relu(output)

    # Layer 4g (Generator)
    # [4096, 128] -> [12288, 64]
    with tf.compat.v1.variable_scope('upconv_4g'):
        output = conv1d_transpose(output, dim, kernel_len, 3)
        output = batchnorm(output)
    output = tf.nn.relu(output)

    # Layer 4d (Discriminator)
    # [4096,1] -> [2048,64]
    with tf.compat.v1.variable_scope('upconv_4d'):
    	y_d = tf.compat.v1.layers.conv1d(y, dim, kernel_len, 2, padding='same')
    y_d = lrelu(y_d)

    # Layer 5
    # [2048, 64]+[12288, 64] -> [14336, 1]
    # [14336, 1] -> [28672, 1]
    with tf.compat.v1.variable_scope('upconv_5'):
        output = tf.concat([y_d,output],1)
        output = conv1d_transpose(output, 1, kernel_len, 2)
    G_z = tf.nn.tanh(output)



  # Automatically update batchnorm moving averages every time G is used during training
    if train and use_batchnorm:
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, scope=tf.compat.v1.get_variable_scope().name)
        assert len(update_ops) == 12
        with tf.control_dependencies(update_ops):
            G_z = tf.identity(G_z)

    # Post Processing
    with tf.compat.v1.variable_scope('filer'):
        G_z = tf.compat.v1.layers.conv1d(G_z, 1, 512, use_bias=False, padding='same')

    return G_z


def lrelu(inputs, alpha=0.2):
    return tf.maximum(alpha * inputs, inputs)


def apply_phaseshuffle(x, rad, pad_type='reflect'):
    b, x_len, nch = x.get_shape().as_list()

    phase = tf.random.uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r
    x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

    x = x[:, phase_start:phase_start+x_len]
    x.set_shape([b, x_len, nch])

    return x


"""
  Input: [None, 32768, nch]
  Output: [None] (linear output)
"""

def WaveGANDiscriminator(x,kernel_len=25,dim=64,use_batchnorm=False,phaseshuffle_rad=0):
    batch_size = tf.shape(x)[0]
    slice_len = int(x.get_shape()[1])
    if use_batchnorm:
        batchnorm = lambda x: tf.compat.v1.layers.batch_normalization(x, training=True)
    else:
        batchnorm = lambda x: x
    if phaseshuffle_rad > 0:
        phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
    else:
        phaseshuffle = lambda x: x

    # Layer 0
    # [32768, 1] -> [8192, 64]
    output = x
    with tf.compat.v1.variable_scope('downconv_0'):
    	output = tf.compat.v1.layers.conv1d(output, dim, kernel_len, 4, padding='same')
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 1
    # [8192, 64] -> [2048, 128]
    with tf.compat.v1.variable_scope('downconv_1'):
    	output = tf.compat.v1.layers.conv1d(output, dim * 2, kernel_len, 4, padding='same')
    	output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 2
    # [2048, 128] -> [512, 256]
    with tf.compat.v1.variable_scope('downconv_2'):
    	output = tf.compat.v1.layers.conv1d(output, dim * 4, kernel_len, 4, padding='same')
    	output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 3
    # [512, 256] -> [128, 512]
    with tf.compat.v1.variable_scope('downconv_3'):
    	output = tf.compat.v1.layers.conv1d(output, dim * 8, kernel_len, 4, padding='same')
    	output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

  	# Layer 4
  	# [128, 512] -> [32, 1024]
    with tf.compat.v1.variable_scope('downconv_4'):
    	output = tf.compat.v1.layers.conv1d(output, dim * 16, kernel_len, 4, padding='same')
    	output = batchnorm(output)
    output = lrelu(output)


    # Layer 5
    # [32, 1024] -> [16, 2048]
    with tf.compat.v1.variable_scope('downconv_5'):
      	output = tf.compat.v1.layers.conv1d(output, dim * 32, kernel_len, 2, padding='same')
    output = batchnorm(output)
    output = lrelu(output)

    # Flatten
    output = tf.reshape(output, [batch_size, -1])

    # Connect to single logit
    with tf.compat.v1.variable_scope('output'):
        output = tf.compat.v1.layers.dense(output, 1)[:, 0]

    # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training
    return output
