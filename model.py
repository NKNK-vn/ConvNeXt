from re import L
import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, LayerNormalization, Activation, GlobalAveragePooling2D, ZeroPadding2D, Add
from keras.activations import gelu
class ConvNeXt:
    def __init__(self, img_size, num_classes = 2, blocks = [3, 3, 9, 3], channel = [96, 192, 384, 768]):
        self.img_size = img_size
        self.num_classes = num_classes
        self.blocks = blocks
        self.channel = channel
    
    def Block(self, filter, x):
        input = x
        x = DepthwiseConv2D(kernel_size=7, padding='same')(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        # x = Conv2D(filters = filter * 4, kernel_size=1 ,padding='same')(x)
        x = Dense(filter*4)(x)
        x = Activation('gelu')(x)
        # x = Conv2D(filters = filter, kernel_size=1 ,padding='same')(x)
        x = Dense(filter)(x)
        x = Add()([input, x])
        return x
        
    def Downsampling(self):   
        downsample = []
        stem = [Conv2D(filters = self.channel[0], kernel_size=4, strides=4, name='stem'), LayerNormalization(epsilon=1e-6)]
        downsample.append(stem)
        for i in range(1, 4):
            downsample.append([LayerNormalization(epsilon=1e-6), Conv2D(filters = self.channel[i], kernel_size=2, strides=2, name='downsample_block_{}'.format(i+1))])
        return downsample
    
    def build_model(self):
        input = Input((self.img_size, self.img_size, 3))
        downsample = self.Downsampling()
        #Block1
        x = downsample[0][0](input)
        x = downsample[0][1](x)
        for i in range(self.blocks[0]):
            x = self.Block(self.channel[0], x)
        #Block2 > 4
        for i in range(1, 4):
            x = downsample[i][0](x)
            x = downsample[i][1](x)
            for _ in range(self.blocks[i]):
                x = self.Block(self.channel[i], x)
        #Fully connected
        x = GlobalAveragePooling2D()(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dense(units=self.num_classes, activation='softmax')(x)
        model = Model(input, x)
        return model
        
if __name__ == '__main__':
    model = ConvNeXt(224)
    print(model.build_model().summary())