# coding: utf-8
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout, Reshape, Lambda, Concatenate, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import VGG16
import tensorflow as tf

def vgg16unet(img_height, img_width, nclasses):
    def bilinear_upsample(image_tensor):
        upsampled = tf.image.resize(image_tensor, size=(img_height, img_width))
        return upsampled
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    # Congeler les poids de toutes les couches de VGG16
    for layer in base_model.layers:
        layer.trainable = False
        
    img_input = Input(shape=(img_height, img_width, 3))
    # Utiliser la sortie de la dernière couche de ResNet50 comme encodeur
    encoder_output = base_model.output
    
    x = Lambda(bilinear_upsample, name='bilinear_upsample')(encoder_output)
    x = Reshape((img_height*img_width, nclasses))(x)
    x = Activation('softmax', name='final_softmax')(x)
    model = Model(inputs=img_input, outputs=x, name='resnet50')
    print('. . . . .Building network successful. . . . .')
    return model

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = UpSampling2D(size=(2, 2))(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_vgg16_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained VGG16 Model """
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = vgg16.get_layer("block1_conv2").output         ## (512 x 512)
    s2 = vgg16.get_layer("block2_conv2").output         ## (256 x 256)
    s3 = vgg16.get_layer("block3_conv3").output         ## (128 x 128)
    s4 = vgg16.get_layer("block4_conv3").output         ## (64 x 64)

    """ Bridge """
    b1 = vgg16.get_layer("block5_conv3").output         ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 256)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 128)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 64)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 32)                      ## (512 x 512)

    """ Output """
    outputs = Conv2D(8, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="VGG16_U-Net")
    return model