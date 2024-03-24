from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.layers import Input, MaxPooling2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout, Softmax
from tensorflow.keras.models import Model


def Conv_BN_Relu(filters, kernel_size, strides, input_layer):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


# ResNet18网络对应的残差模块a和残差模块b
def resiidual_a_or_b(input_x, filters, flag):
    if flag == 'a':
        # 主路
        x = Conv_BN_Relu(filters, (3, 3), 1, input_x)
        x = Conv_BN_Relu(filters, (3, 3), 1, x)

        # 输出
        y = Add()([x, input_x])

        return y
    elif flag == 'b':
        # 主路
        x = Conv_BN_Relu(filters, (3, 3), 2, input_x)
        x = Conv_BN_Relu(filters, (3, 3), 1, x)

        # 支路下采样
        input_x = Conv_BN_Relu(filters, (1, 1), 2, input_x)

        # 输出
        y = Add()([x, input_x])

        return y

def get_resnet18(x):

    # 第一层
    input_img = Input(shape=x.shape[1:])
    conv1 = Conv_BN_Relu(64, (7, 7), 1, input_img)
    conv1_Maxpooling = MaxPooling2D((3, 3), strides=2, padding='same')(conv1)

    # conv2_x
    x = resiidual_a_or_b(conv1_Maxpooling, 64, 'b')
    x = resiidual_a_or_b(x, 64, 'a')

    # conv3_x
    x = resiidual_a_or_b(x, 128, 'b')
    x = resiidual_a_or_b(x, 128, 'a')

    # conv4_x
    x = resiidual_a_or_b(x, 256, 'b')
    x = resiidual_a_or_b(x, 256, 'a')

    # conv5_x
    x = resiidual_a_or_b(x, 512, 'b')
    x = resiidual_a_or_b(x, 512, 'a')

    # 最后一层
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    model_out = Dense(512)(x)

    model = Model(input_img, model_out)

    return model
