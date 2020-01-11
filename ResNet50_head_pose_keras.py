# -*- coding: utf-8 -*-
"""
Created on Mon Jan 06 20:33:32 2020
<<<<<<< HEAD
=======

>>>>>>> a778c65b3e0840cab83e18cdd839e092c19f4835
@author: Ardhendu
"""

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input as res50_pp_input
import numpy as np
from keras import layers, models 
from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from custom_validate_callback import CustomCallback
from os.path import dirname, realpath

from keras_self_attention import SeqSelfAttention
from keras_self_attention import SeqWeightedAttention as Attention

import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def resnet_block5_weights(model, tensor_b1):
    #tensor_b1 = layers.Input(shape=(14, 14, 1024))
    #base_out = layers.Input(shape=(14, 14, 1024))
    x = conv_block(tensor_b1, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    
    ''' The output of x (7,7,2048). Add a pooling layer: experiment with global 
    pooling and local pooling with (max and average option) with flatten in the 
    end to feed to Attention
    '''
    x = layers.GlobalMaxPooling2D()(x)
    res_block5 = Model(inputs = tensor_b1, outputs = x)
    #res_block5.summary()
    
    ''' Set the weights from the original network '''
    for i in range(1, len(res_block5.layers)-1): #extra pooling layer in res_block5
        j = i + base_layer
        weights = model.layers[j].get_weights()
        #print(model.layers[j].name)
        if len(weights) == 0:
            continue
        res_block5.layers[i].set_weights(weights)
        #print(res_block5.layers[i].name)
    return res_block5


"""Constants"""

input_tensor = layers.Input(shape=(224, 224, 3))

is_regression = False #define whether regression/classification should be used
#root_dir = 'E:/AndrewGidney/FaceOrientation'
train_data_dir = '{}/{}/train/'
test_data_dir = '{}/{}/test/'
val_data_dir = '{}/{}/val/'
base_model_dir = '{}/BaseModels/'
output_model_dir = '{}/TrainedModels/'
metrics_dir = '{}/Metrics/'
output_model_filename = '{}.h5'.format('resnet50')
training_metrics_filename = output_model_filename + '(Training).csv'
image_size = (224,224) #image resolution in pixels
nb_train_samples = 10000
nb_test_samples = 0 #number of images used for testing
nb_val_samples = 5000
verbose = 1
validation_steps = 5 #number of epochs between validation
csv_logger = CSVLogger(metrics_dir + training_metrics_filename)
optimizer = Adam()

checkpointer = ModelCheckpoint(filepath = output_model_dir + output_model_filename + '.{epoch:02d}.h5', verbose=1, save_weights_only=False, period=validation_steps)


loss_type = 'mean_absolute_error'
metrics = ['mae']

'''Just before the block5 to capture the output'''
base_layer = 140 

model = ResNet50(weights='imagenet', input_tensor=input_tensor, include_top=False)


base_out = model.layers[base_layer].output #tapped output before block5

tensor_yaw = layers.Input(shape=(14, 14, 1024)) #Input to the parallal stream
tensor_pitch = layers.Input(shape=(14, 14, 1024)) #Input to the parallal stream
tensor_roll = layers.Input(shape=(14, 14, 1024)) #Input to the parallal stream
yaw = resnet_block5_weights(model,tensor_yaw)
pitch = resnet_block5_weights(model,tensor_pitch)
roll = resnet_block5_weights(model,tensor_roll)

x = yaw(base_out)
x = layers.Dense(1, activation='tanh', name="Yaw_output")(x)

y = pitch(base_out)
y = layers.Dense(1, activation='tanh', name="Pitch_output")(y)

z = roll(base_out)
z = layers.Dense(1, activation='tanh', name="Roll_output")(z)

'''
# define dictionaries for the specified loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
'''

losses = {
	"Yaw_output": "mean_absolute_error",
	"Pitch_output": "mean_absolute_error",
    "Roll_output": "mean_absolute_error"
}
lossWeights = {"Yaw_output": 1.0, "Pitch_output": 1.0, "Roll_output": 1.0}

model = Model(inputs=model.input, outputs=[x,y,z])

model.compile(loss=loss_type, loss_weights=lossWeights, metrics=metrics, optimizer=optimizer)
<<<<<<< HEAD
   
=======
   

>>>>>>> a778c65b3e0840cab83e18cdd839e092c19f4835
