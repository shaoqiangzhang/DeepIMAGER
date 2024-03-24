from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.layers import Input, MaxPooling2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout, Softmax
from tensorflow.keras.models import Model


def Conv_BN_Relu(filters, kernel_size, strides, input_layer):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def resiidual_c_or_d(input_x, filters, flag):
    if flag == 'c':
        # 主路
        x = Conv_BN_Relu(filters, (1, 1), 1, input_x)
        x = Conv_BN_Relu(filters, (3, 3), 1, x)
        x = Conv_BN_Relu(filters * 4, (1, 1), 1, x)

        # 输出
        y = Add()([x, input_x])

        return y
    elif flag == 'd':
        # 主路
        x = Conv_BN_Relu(filters, (1, 1), 2, input_x)
        x = Conv_BN_Relu(filters, (3, 3), 1, x)
        x = Conv_BN_Relu(filters * 4, (1, 1), 1, x)

        # 支路下采样
        input_x = Conv_BN_Relu(filters * 4, (1, 1), 2, input_x)

        # 输出
        y = Add()([x, input_x])

        return y



