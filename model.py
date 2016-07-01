import tensorflow as tf
import numpy as np
import sys
from network import *

class Model:
    @staticmethod 
    def vggnet(_X, _dropout):
        # TODO weight decay loss tern
        # Layer 1 (conv-relu-pool)
        conv1_1 = conv(_X, 3, 3, 64, 1, 1, padding='SAME', name='conv1_1')
        conv1_2 = conv(conv1_1, 3, 3, 64, 1, 1, padding='SAME', name='conv1_2')
        pool1 = max_pool(conv1_2, 2, 2, 2, 2, padding='SAME', name='pool1')
        # Layer 2 (conv-relu-pool)
        conv2_1 = conv(pool1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_1')
        conv2_2 = conv(conv2_1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_2')
        pool2 = max_pool(conv2_2, 2, 2, 2, 2, padding='SAME', name='pool2')
        # Layer 3 (conv-relu-pool)
        conv3_1 = conv(pool2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_1')
        conv3_2 = conv(conv3_1, 3, 3, 256, 1, 1, padding='SAME', name='conv3_2')
        conv3_3 = conv(conv3_2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_3')
        pool3 = max_pool(conv3_3, 2, 2, 2, 2, padding='SAME', name='pool3')
        # Layer 4 (conv-relu-pool)
        conv4_1 = conv(pool3, 3, 3, 512, 1, 1, padding='SAME', name='conv4_1')
        conv4_2 = conv(conv4_1, 3, 3, 512, 1, 1, padding='SAME', name='conv4_2')
        conv4_3 = conv(conv4_2, 3, 3, 512, 1, 1, padding='SAME', name='conv4_3')
        pool4 = max_pool(conv4_3, 2, 2, 2, 2, padding='SAME', name='pool4')
        # Layer 5 (conv-relu-pool)
        conv5_1 = conv(pool4, 3, 3, 512, 1, 1, padding='SAME', name='conv5_1')
        conv5_2 = conv(conv5_1, 3, 3, 512, 1, 1, padding='SAME', name='conv5_2')
        conv5_3 = conv(conv5_2, 3, 3, 512, 1, 1, padding='SAME', name='conv5_3')
        pool5 = max_pool(conv5_3, 2, 2, 2, 2, padding='SAME', name='pool5')
        
        # Layer 6 (fc-relu-drop)
        fc6 = tf.reshape(pool5, [-1, 7*7*512])
        fc6 = fc(fc6, 7*7*512, 4096, name='fc6')
        fc6 = dropout(fc6, _dropout)
        # Layer 7 (fc-relu-drop)
        fc7 = fc(fc6, 4096, 4096, name='fc7')
        fc7 = dropout(fc7, _dropout)
        # Layer 8 (fc-prob)
        fc8 = fc(fc7, 4096, 10, relu=False, name='fc8')
        return fc8
        
        
#class VGG_ILSVRC_16_layers(Network):
#    def setup(self):
#        (self.feed('data')
#             .conv(3, 3, 64, 1, 1, name='conv1_1')
#             .conv(3, 3, 64, 1, 1, name='conv1_2')
#             .max_pool(2, 2, 2, 2, name='pool1')
#             .conv(3, 3, 128, 1, 1, name='conv2_1')
#             .conv(3, 3, 128, 1, 1, name='conv2_2')
#             .max_pool(2, 2, 2, 2, name='pool2')
#             .conv(3, 3, 256, 1, 1, name='conv3_1')
#             .conv(3, 3, 256, 1, 1, name='conv3_2')
#             .conv(3, 3, 256, 1, 1, name='conv3_3')
#             .max_pool(2, 2, 2, 2, name='pool3')
#             .conv(3, 3, 512, 1, 1, name='conv4_1')
#             .conv(3, 3, 512, 1, 1, name='conv4_2')
#             .conv(3, 3, 512, 1, 1, name='conv4_3')
#             .max_pool(2, 2, 2, 2, name='pool4')
#             .conv(3, 3, 512, 1, 1, name='conv5_1')
#             .conv(3, 3, 512, 1, 1, name='conv5_2')
#             .conv(3, 3, 512, 1, 1, name='conv5_3')
#             .max_pool(2, 2, 2, 2, name='pool5')
#             .fc(4096, name='fc6')
#             .fc(4096, name='fc7')
#             .fc(1000, relu=False, name='fc8')
#             .softmax(name='prob'))