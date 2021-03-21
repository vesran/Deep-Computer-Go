import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import golois


##################################################################
# Parameters
##################################################################

PLANES = 21
MOVES = 361
N = 100_000
DIM = 19


##################################################################
# Blocks
##################################################################

def shaping_data():
    input_data = np.random.randint(2, size=(N, DIM, DIM, PLANES))
    input_data = input_data.astype('float32')

    policy = np.random.randint(MOVES, size=(N,))
    policy = keras.utils.to_categorical(policy)

    value = np.random.randint(2, size=(N,))
    value = value.astype('float32')

    end = np.random.randint(2, size=(N, DIM, DIM, 2))
    end = end.astype('float32')

    groups = np.zeros((N, DIM, DIM, 1))
    groups = groups.astype('float32')

    return input_data, policy, value, end, groups


def hswish(x):
    return x * tf.nn.relu6(x+3) * 0.166666666667


def se_block(in_block, ch, ratio=16):
    x = layers.Dropout(0.2)(in_block)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(ch//ratio, activation='relu')(x)
    x = layers.Dense(ch, activation='sigmoid')(x)
    return layers.Multiply()([in_block, x])



def channel_attention_module(in_block, filters, ratio):
    maxp = layers.GlobalMaxPooling2D()(in_block)
    avgp = layers.GlobalAveragePooling2D()(in_block)

    hidden_ff = layers.Dense(filters // ratio, activation='relu')
    out_ff = layers.Dense(filters)

    maxp = hidden_ff(maxp)
    maxp = out_ff(maxp)

    avgp = hidden_ff(avgp)
    avgp = out_ff(avgp)

    add_x = layers.add([maxp, avgp])
    activ_x = layers.Activation('sigmoid')(add_x)
    return layers.Multiply()([in_block, activ_x])


def spatial_attention_module(in_block):
    maxp = layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(in_block)
    avgp = layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(in_block)

    max_avg = layers.Concatenate()([maxp, avgp])

    conv_x = layers.Conv2D(1, (7, 7), padding='same')(max_avg)

    activ_x = layers.Activation('sigmoid')(conv_x)
    return layers.Multiply()([in_block, activ_x])


def residual_block(x, filters):
    x1 = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = hswish(x1)

    x1 = layers.Conv2D(filters, (3, 3), padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)

    x1 = se_block(x1, filters, ratio=4)

    x = layers.add([x1, x])
    x = hswish(x)
    x = layers.BatchNormalization()(x)

    return x


def input_block(filters, inp):
    x = layers.Conv2D(filters, (3, 3), padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = hswish(x)
    x1 = layers.Conv2D(filters, (5, 5), padding='same')(inp)
    x1 = layers.BatchNormalization()(x1)
    x1 = hswish(x1)
    x = layers.add([x, x1])
    return x


def output_policy_block(policy_head):
    policy_head = layers.Conv2D(1, 1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.0001))(
        policy_head)
    policy_head = hswish(policy_head)
    policy_head = layers.BatchNormalization()(policy_head)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Activation('softmax', name='policy')(policy_head)
    return policy_head


def output_value_block(value_head):
    value_head = layers.GlobalAveragePooling2D()(value_head)
    value_head = layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001))(value_head)
    value_head = hswish(value_head)
    value_head = layers.BatchNormalization()(value_head)
    value_head = layers.Dropout(0.2)(value_head)
    value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(
        value_head)
    return value_head


def create_model(filters, num_blocks=6):
    inp = tf.keras.Input(shape=(19, 19, PLANES), name='board')
    x = input_block(filters, inp)

    for i in range(num_blocks):
        x = residual_block(x, filters)

    policy_head = output_policy_block(x)
    value_head = output_value_block(x)

    model = keras.Model(inputs=inp, outputs=[policy_head, value_head])
    return model

