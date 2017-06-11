import os
import numpy as np
import tensorflow as tf
import random
from unittest.mock import MagicMock

def _print_success_message():
    print('Tests Passed')

def test_nn_sentence_inputs(neural_net_sentence_input):
    sentence_size = 56
    nn_inputs_out_x = neural_net_sentence_input(sentence_size)

    assert nn_inputs_out_x.get_shape().as_list() == [None, sentence_size],\
        'Incorrect Sentence Shape.  Found {} shape'.format(nn_inputs_out_x.get_shape().as_list())

    assert nn_inputs_out_x.op.type == 'Placeholder',\
        'Incorrect Sentence Type.  Found {} type'.format(nn_inputs_out_x.op.type)

    assert nn_inputs_out_x.name == 'input_x:0', \
        'Incorrect Name.  Found {}'.format(nn_inputs_out_x.name)

    print('Sentence Input Tests Passed.')


def test_nn_label_inputs(neural_net_label_input):
    n_classes = 2
    nn_inputs_out_y = neural_net_label_input(n_classes)

    assert nn_inputs_out_y.get_shape().as_list() == [None, n_classes],\
        'Incorrect Label Shape.  Found {} shape'.format(nn_inputs_out_y.get_shape().as_list())

    assert nn_inputs_out_y.op.type == 'Placeholder',\
        'Incorrect Label Type.  Found {} type'.format(nn_inputs_out_y.op.type)

    assert nn_inputs_out_y.name == 'input_y:0', \
        'Incorrect Name.  Found {}'.format(nn_inputs_out_y.name)

    print('Label Input Tests Passed.')


def test_nn_keep_prob_inputs(neural_net_keep_prob_input):
    nn_inputs_out_k = neural_net_keep_prob_input()

    assert nn_inputs_out_k.get_shape().ndims is None,\
        'Too many dimensions found for keep prob.  Found {} dimensions.  It should be a scalar (0-Dimension Tensor).'.format(nn_inputs_out_k.get_shape().ndims)

    assert nn_inputs_out_k.op.type == 'Placeholder',\
        'Incorrect keep prob Type.  Found {} type'.format(nn_inputs_out_k.op.type)

    assert nn_inputs_out_k.name == 'keep_prob:0', \
        'Incorrect Name.  Found {}'.format(nn_inputs_out_k.name)

    print('Keep Prob Tests Passed.')

def test_embed(embedding_creation):
    vocab_size = 20000
    embedding_size = 128
    sequence_length = 56
    channels = 1
    input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
    embed = embedding_creation(input_x,vocab_size,embedding_size)

    assert embed.get_shape().as_list() == [None, sequence_length, embedding_size, channels],\
        'Incorrect Shape.  Found {} shape'.format(embed.get_shape().as_list())

    _print_success_message()

def test_con_pool(conv2d_maxpool):
    #sentence length, embbed size, number channels
    test_x = tf.placeholder(tf.float32, [None, 56, 128,1])
    test_num_filters = 3
    test_filter_size = 3

    conv2d_maxpool_out = conv2d_maxpool(test_x, test_num_filters, test_filter_size)

    assert conv2d_maxpool_out.get_shape().as_list() == [None, 1, 1, test_num_filters],\
        'Incorrect Shape.  Found {} shape'.format(conv2d_maxpool_out.get_shape().as_list())

    _print_success_message()

def test_apply_filters(apply_conv_filters,conv2d_maxpool):
    filter_sizes = [3,4,5]
    num_filters = 128
    test_x = tf.placeholder(tf.float32, [None, 56, 128,1])
    pool = apply_conv_filters(test_x,filter_sizes,num_filters)
    num_filters_total = num_filters * len (filter_sizes)

    assert pool.get_shape().as_list()[3] == num_filters_total,\
        'Incorrect Filters Concatenation.  Found {} concat'.format(pool.get_shape().as_list()[3])

    assert pool.get_shape().as_list() == [None, 1, 1, num_filters_total],\
        'Incorrect Shape.  Found {} shape'.format(pool.get_shape().as_list())

    _print_success_message()

def test_flatten(flatten):
    test_x = tf.placeholder(tf.float32, [None, 10, 30, 6])
    flat_out = flatten(test_x)

    assert flat_out.get_shape().as_list() == [None, 10*30*6],\
        'Incorrect Shape.  Found {} shape'.format(flat_out.get_shape().as_list())

    _print_success_message()


def test_compute_output(calculate_output):
    test_x = tf.placeholder(tf.float32, [None, 128])
    test_num_outputs = 40
    fc_out, predictions = calculate_output(test_x, test_num_outputs)

    assert fc_out.get_shape().as_list() == [None, 40],\
        'Incorrect Shape.  Found {} shape'.format(fc_out.get_shape().as_list())

    _print_success_message()

















