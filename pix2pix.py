from argparse import ArgumentParser, ArgumentTypeError
from collections import namedtuple
from glob import glob
from natsort import natsorted
import numpy as np
import os
import tensorflow as tf


ImageSet = namedtuple("ImageSet",
    ["paths", "inputs", "targets", "count", "steps_per_epoch"])

Model = namedtuple("Model",
    ["outputs", "predict_real", "predict_fake", "discrim_loss",
     "discrim_grads_and_vars", "gen_loss_GAN", "gen_loss_L1",
     "gen_grads_and_vars", "train"])     


def preprocess_image(tensor):
    with tf.name_scope("preprocess_image"):
        return -1+2*tf.image.convert_image_dtype(tensor, dtype=tf.float32)

def deprocess_image(tensor, dtype=tf.uint16):
    with tf.name_scope("deprocess_image"):
        return tf.image.convert_image_dtype((tensor+1)/2, dtype=dtype, saturate=True)    

def decode_image(tensor, channels, dtype=tf.uint16):
    with tf.name_scope("decode_image"):
        return preprocess_image(tf.image.decode_png(tensor, channels=channels, dtype=dtype))

def encode_image(tensor):
    with tf.name_scope("encode_image"):
        return tf.map_fn(tf.image.encode_png, deprocess_image(tensor), dtype=tf.string)        


def gen_conv(batch_inputs, filters):
    kernel_initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(batch_inputs,
        filters=filters,
        kernel_size=4,
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer)

def gen_deconv(batch_inputs, filters):
    kernel_initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(batch_inputs,
        filters=filters,
        kernel_size=4,
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer)

def discrim_conv(batch_inputs, filters, stride):
    padded_inputs = tf.pad(batch_inputs, [[0, 0], [1, 1], [1, 1], [0, 0]])
    kernel_initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(padded_inputs,
        filters=filters,
        kernel_size=4,
        strides=(stride, stride),
        padding="valid",
        kernel_initializer=kernel_initializer)    

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5*(1+a))*x + (0.5*(1-a))*tf.abs(x)

def batchnorm(inputs):
    gamma_initializer = tf.random_normal_initializer(1.0, 0.02)
    return tf.layers.batch_normalization(inputs,
        axis=3,
        epsilon=1e-5,
        momentum=0.1,
        training=True,
        gamma_initializer=gamma_initializer)


def load_images(input_dir, direction, batch_size, image_size, image_channels, shuffle=False):
    assert os.path.exists(input_dir), "Input directory does not exist"

    input_paths = glob(os.path.join(input_dir, "*.png"))
    assert len(input_paths) > 0, "Input directory contains no image files (*.png)"
    input_paths = natsorted(input_paths)

    with tf.name_scope("load_images"):
        queue = tf.train.string_input_producer(input_paths, shuffle=shuffle)
        reader = tf.WholeFileReader()
        paths, contents = reader.read(queue)
        images = decode_image(contents, channels=image_channels)
        a_images = images[:,:image_size,:]
        b_images = images[:,image_size:,:]
        a_images.set_shape([image_size, image_size, image_channels])
        b_images.set_shape([image_size, image_size, image_channels])            

    if direction == "AB":
        with tf.name_scope("input_images"):
            input_images = tf.identity(a_images)
        with tf.name_scope("target_images"):
            target_images = tf.identity(b_images)
    elif direction == "BA":
        with tf.name_scope("input_images"):
            input_images = tf.identity(b_images)
        with tf.name_scope("target_images"):
            target_images = tf.identity(a_images)
    else:
        raise Exception("Direction must be 'AB' or 'BA'")     

    paths_batch, inputs_batch, targets_batch = tf.train.batch(
        [paths, input_images, target_images], batch_size=batch_size)

    steps_per_epoch = int(np.ceil(len(input_paths)/batch_size))

    return ImageSet(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch)

def save_images(fetches, output_dir, step=None,
                kind=["inputs", "outputs", "targets"]):
    assert os.path.exists(output_dir), "Ouput directory does not exist"

    filesets = []
    for i, input_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(input_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for which in kind:
            filename = "%s-%s.png" % (name, which)
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[which] = filename
            output_path = os.path.join(output_dir, filename)
            content = fetches[which][i]
            with open(output_path, "wb") as f:
                f.write(content)
        filesets.append(fileset)
    return filesets    


def create_generator(gen_inputs, output_channels, min_filters=64):
    layers = []

    input_shape = gen_inputs.get_shape().as_list()
    assert input_shape[1] == input_shape[2]
    depth = int(np.log2(input_shape[1]))

    with tf.variable_scope("encoder_1"):
        inputs = gen_inputs
        outputs = gen_conv(inputs, min_filters)
        layers.append(outputs)

    n = depth - 1
    for layer in range(n):
        with tf.variable_scope("encoder_%d" % (len(layers)+1)):
            inputs = layers[-1]
            outputs = lrelu(inputs, 0.2)
            filters = min_filters*min(2**(layer+1), 8)
            outputs = gen_conv(outputs, filters)
            outputs = batchnorm(outputs)
            layers.append(outputs)

    for layer in range(n):
        skip_layer = n-layer
        with tf.variable_scope("decoder_%d" % (skip_layer+1)):
            if layer == 0:
                inputs = layers[-1] # first layer has no skip connection
            else:
                inputs = tf.concat([layers[-1], layers[skip_layer]], axis=3)
            outputs = tf.nn.relu(inputs)
            filters = min_filters*min(2**((n-1)-layer), 8)
            outputs = gen_deconv(outputs, filters)
            outputs = batchnorm(outputs)
            #outputs = tf.nn.dropout(outputs, keep_prob=1-dropout)
            layers.append(outputs)

    with tf.variable_scope("decoder_1"):
        inputs = tf.concat([layers[-1], layers[0]], axis=3)
        outputs = tf.nn.relu(inputs)
        outputs = gen_deconv(outputs, output_channels)
        outputs = tf.tanh(outputs)
        layers.append(outputs)

    return layers[-1]

def create_discriminator(discrim_inputs, discrim_targets, min_filters=64, output_size=6):
    layers = []

    input_shape = discrim_inputs.get_shape().as_list()
    assert input_shape[1] == input_shape[2]
    depth = int(np.log2(input_shape[1]/(output_size+2)))
    assert depth > 0

    with tf.variable_scope("layer_1"):
        inputs = tf.concat([discrim_inputs, discrim_targets], axis=3)            
        outputs = discrim_conv(inputs, min_filters, 2)
        outputs = lrelu(outputs, 0.2)
        layers.append(outputs)

    n = depth
    for layer in range(n):
        with tf.variable_scope("layer_%d" % (len(layers)+1)):
            inputs = layers[-1]
            filters = min_filters*min(2**(layer+1), 8)
            stride = 1 if layer == n-1 else 2 # last layer has stride 1                
            outputs = discrim_conv(inputs, filters, stride)
            outputs = batchnorm(outputs)
            outputs = lrelu(outputs, 0.2)
            layers.append(outputs)

    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        inputs = layers[-1]
        outputs = discrim_conv(inputs, 1, 1)
        outputs = tf.sigmoid(outputs)
        layers.append(outputs)

    return layers[-1]    


def create_model(model_inputs, model_targets, gan_weight=1, l1_weight=100,
                 lr=0.0002, beta1=0.5, eps=1e-12):

    with tf.variable_scope("generator"):
        output_channels = model_targets.get_shape().as_list()[-1]
        gen_outputs = create_generator(model_inputs, output_channels)

    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            predict_real = create_discriminator(model_inputs, model_targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            predict_fake = create_discriminator(model_inputs, gen_outputs)

    with tf.name_scope("discriminator_loss"):
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real+eps) + tf.log(1-predict_fake+eps)))

    with tf.name_scope("generator_loss"):
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake+eps))
        gen_loss_L1 = tf.reduce_mean(tf.abs(model_targets - gen_outputs))
        gen_loss = gen_loss_GAN*gan_weight + gen_loss_L1*l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables()
            if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables()
                if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(lr, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=gen_outputs,
        train=tf.group(update_losses, incr_global_step, gen_train))

def restore_session(sess, saver, checkpoint):
    tf.logging.info("Loading model from checkpoint")
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint)
    saver.restore(sess, latest_checkpoint)

def run(input_dir, output_dir, mode, direction, batch_size, image_size,
        image_channels, epochs, checkpoint, summary_freq, logging_freq,
        save_freq, logging_verbosity, seed):

    if seed is not None:
        tf.set_random_seed(seed)

    if logging_verbosity is not None:
        tf.logging.set_verbosity(logging_verbosity)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  

    examples = load_images(input_dir,
                           direction=direction,
                           batch_size=batch_size,
                           image_size=image_size,
                           image_channels=image_channels,
                           shuffle=(mode == "train"))

    tf.logging.info("Loaded %d images" % examples.count)

    model = create_model(examples.inputs, examples.targets)

    with tf.name_scope("output_images"):
        encoded_inputs = encode_image(examples.inputs)
        encoded_targets = encode_image(examples.targets)
        encoded_outputs = encode_image(model.outputs)        
    
    # summary images
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", deprocess_image(examples.inputs, dtype=tf.uint8))
    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", deprocess_image(examples.targets, dtype=tf.uint8))
    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", deprocess_image(model.outputs, dtype=tf.uint8))
    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", deprocess_image(model.predict_real, dtype=tf.uint8))
    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", deprocess_image(model.predict_fake, dtype=tf.uint8))

    # summary scalars
    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    # summary histograms
    for var in tf.trainable_variables():
        tf.summary.histogram("%s/values" % var.op.name, var)
    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram("%s/gradients" % var.op.name, grad)

    merged_summary = tf.summary.merge_all()

    #with tf.name_scope("parameter_count"):
    #    parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(var))
    #        for var in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    if mode == "train":
        assert epochs is not None, "'epochs' is a required parameter"

        training_hooks = []

        training_hooks.append(
            tf.train.StopAtStepHook(
                last_step=examples.steps_per_epoch*epochs))

        if summary_freq is not None:
            training_hooks.append(
                tf.train.SummarySaverHook(
                    output_dir=output_dir,
                    save_steps=summary_freq,
                    summary_op=merged_summary))

        if save_freq is not None:
            training_hooks.append(
                tf.train.CheckpointSaverHook(
                    checkpoint_dir=output_dir,
                    save_steps=save_freq,
                    saver=saver))

        if logging_freq is not None:
            training_hooks.append(
                tf.train.LoggingTensorHook(
                    tensors={
                        "global_step": tf.train.get_global_step(),
                        "discrim_loss": model.discrim_loss,
                        "gen_loss_GAN": model.gen_loss_GAN,
                        "gen_loss_L1": model.gen_loss_L1},
                    every_n_iter=logging_freq))

        with tf.train.MonitoredSession(hooks=training_hooks) as sess:
            if checkpoint is not None:
                restore_session(sess, saver, checkpoint)

            while not sess.should_stop():
                sess.run(model.train)

    elif mode == "test":
        assert checkpoint is not None, "'checkpoint' is a required parameter"
        
        with tf.train.MonitoredSession() as sess:
            restore_session(sess, saver, checkpoint)

            for step in range(examples.steps_per_epoch):
                fetches = {
                    "paths": examples.paths,
                    "inputs": encoded_inputs,
                    "targets": encoded_targets,
                    "outputs": encoded_outputs}
                results = sess.run(fetches)
                save_images(results, output_dir)

    else:
        raise Exception("Mode must be 'train' or 'test'")


def type_power2(string):
    value = int(string)
    if not (np.ceil(np.log2(value)) == np.floor(np.log2(value))):
        raise ArgumentTypeError("Argument must be a power of 2")
    return value

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", required=True, choices=["train", "test"])
    parser.add_argument("--direction", default="AB", choices=["AB", "BA"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=type_power2, default=32)
    parser.add_argument("--image_channels", type=int, default=1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--summary_freq", type=int)
    parser.add_argument("--logging_freq", type=int)
    parser.add_argument("--save_freq", type=int)
    parser.add_argument("--logging_verbosity", default="INFO")    
    parser.add_argument("--seed", type=int)
    return parser.parse_args()

if __name__ == "__main__":
    opts = parse_arguments()
    run(**vars(opts))
