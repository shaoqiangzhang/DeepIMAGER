from __future__ import print_function

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse

import sys

parser = argparse.ArgumentParser(description="example")

#parser.add_argument('-node_num',help='')

#args = parser.parse_args()
from numba import jit, cuda
import tensorflow.keras as keras
import tensorflow as tf


# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
# #设置GPU定量分配
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
#
#
#
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# #from keras import backend as K
# #K._get_available_gpus()
#
#
# #设置GPU按需分配
# config = tf.compat.v1.ConfigProto( device_count = {'GPU': 0} )
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)



from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import os,sys
import numpy as np

from sklearn import metrics
from scipy import interp
import pandas as pd
from model.resnet50 import *

parser = argparse.ArgumentParser(description="")

parser.add_argument('-num_batches', type=int, required=True, default=None, help="Number of TF or the number of x file.")
parser.add_argument('-data_path', required=True, default=None, help="The path that includes x file, y file and z file.")
parser.add_argument('-output_dir', required=True, default="./output/", help="Indicate the path for output.")

parser.add_argument('-weight_path', default=None, help="The path for a trained model.")

args = parser.parse_args()


class use_model_predict:
    def __init__(self, num_batches=5, output_dir=None, data_path=None, predict_output_dir=None):
        # ###################################### parameter settings
        self.data_augmentation = False
        # num_predictions = 20
        self.batch_size = 32  # mini batch for training
        # num_classes = 3   # ### categories of labels
        #self.epochs = 200  # ### iterations of trainning, with GPU 1080, 200 for KEGG and Reactome, depends on specific tasks for GTRD, we actually selected
        # the best epochs and learning rate by a test on the first three TF in list
        # length_TF =3057  # number of divide data parts
        # num_predictions = 20
        self.epochs = 200
        self.model_name = 'keras_resnet_trained_model_DeepDRIM.h5'
        self.output_dir = output_dir
        if output_dir is not None:
            if not os.path.isdir(self.output_dir):
                os.mkdir(self.output_dir)
        #####
        self.num_batches = num_batches  # number of data parts divided
        self.data_path = data_path
        self.num_classes = 2
        self.whole_data_TF = [i for i in range(self.num_batches)]
        ###################################################
        self.x_train = None
        self.y_train = None
        self.z_train = None
        self.count_set_train = None
        self.x_test = None
        self.y_test = None
        self.z_test = None
        self.count_set = None
        self.load_model_path = None
        self.predict_output_dir = predict_output_dir


    def load_data_TF2(self,indel_list,data_path, num_of_pair_ratio=1):  # cell type specific  ## random samples for reactome is not enough, need borrow some from keggp
        import random
        import numpy as np
        xxdata_list = []
        yydata = []
        zzdata = []
        count_set = [0]
        count_setx = 0
        for i in indel_list:#len(h_tf_sc)):
            xdata = np.load(data_path+str(i)+'_xdata.npy')
            ydata = np.load(data_path+str(i)+'_ydata.npy')
            zdata = np.load(data_path+str(i)+'_zdata.npy')

            num_of_pairs = round(num_of_pair_ratio*len(ydata))
            all_k_list = list(range(len(ydata)))
            select_k_list = all_k_list[0:num_of_pairs]

            for k in select_k_list:
                xxdata_list.append(xdata[k,:,:,:])
                yydata.append(ydata[k])
                zzdata.append(zdata[k])
            count_setx = count_setx + len(ydata)
            count_set.append(count_setx)
            print (i,len(ydata))
        yydata_array = np.array(yydata)
        yydata_x = yydata_array.astype('int')
        print(np.array(xxdata_list).shape)
        return((np.array(xxdata_list),yydata_x,count_set,np.array(zzdata)))


    def update_test_train_data(self, test_indel,epochs,num_of_pair_ratio=1):
        print("len test_indel",test_indel)
        if type(test_indel)!=list:
            test_TF = [test_indel]  #
        else:
            test_TF = test_indel
        train_TF = [i for i in self.whole_data_TF if i not in test_TF]  #
        #####################################################################
        (self.x_train, self.y_train, self.count_set_train,self.z_train) = self.load_data_TF2(train_TF, self.data_path,num_of_pair_ratio)
        (self.x_test, self.y_test, self.count_set,self.z_test) = self.load_data_TF2(test_TF, self.data_path,num_of_pair_ratio)
        print(self.x_train.shape, 'x_train samples')
        print(self.x_test.shape, 'x_test samples')
        if self.output_dir is not None:
            self.save_dir = os.path.join(self.output_dir, str(test_indel) + '_saved_models' + str(epochs))  ## the result folder
        else:
            self.save_dir="."
        if self.num_classes > 2:
            self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
            self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        print(self.y_train.shape, 'y_train samples')
        print(self.y_test.shape, 'y_test samples')
        ############
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

    def get_single_image_resnet50(self, x_train):
        print("x_train.shape in single image", x_train.shape)
        # 第一层
        input_img = keras.layers.Input(shape=x_train.shape[1:])
        conv1 = Conv_BN_Relu(64, (7, 7), 1, input_img)
        conv1_Maxpooling = MaxPooling2D((3, 3), strides=2, padding='same')(conv1)
        x = conv1_Maxpooling

        # 中间层
        filters = 64
        num_residuals = [3, 4, 6, 3]
        for i, num_residual in enumerate(num_residuals):
            for j in range(num_residual):
                if j == 0:
                    x = resiidual_c_or_d(x, filters, 'd')
                else:
                    x = resiidual_c_or_d(x, filters, 'c')
            filters = filters * 2

        # 最后一层
        x = GlobalAveragePooling2D()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.5)(x)

        model_out = keras.layers.Dense(512)(x)

        return keras.Model(input_img, model_out)

    def get_pair_image_resnet50(self, x_train):
        print("x_train.shape in multi image", x_train.shape)
        # 第一层
        input_img = keras.layers.Input(shape=x_train.shape[1:])
        conv1 = Conv_BN_Relu(64, (7, 7), 1, input_img)
        conv1_Maxpooling = MaxPooling2D((3, 3), strides=2, padding='same')(conv1)
        x = conv1_Maxpooling

        # 中间层
        filters = 64
        num_residuals = [3, 4, 6, 3]
        for i, num_residual in enumerate(num_residuals):
            for j in range(num_residual):
                if j == 0:
                    x = resiidual_c_or_d(x, filters, 'd')
                else:
                    x = resiidual_c_or_d(x, filters, 'c')
            filters = filters * 2

        # 最后一层
        x = GlobalAveragePooling2D()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.5)(x)

        model_out = keras.layers.Dense(512)(x)

        return keras.Model(input_img, model_out)



    def construct_model(self, x_train):
        ############
        print("x shape", x_train.shape)

        n=x_train.shape[1]
        x1=x_train[:, 0, :, :,np.newaxis]

        x2=x_train[:, 1:n, :, :,np.newaxis]
        x2_1=x2[:,0,:,:,:]


        single_image_model = self.get_single_image_resnet50(x1)
        input_img_single = keras.layers.Input(shape=x1.shape[1:])
        single_image_out = single_image_model(input_img_single)

        pair_image_model = self.get_pair_image_resnet50(x2_1)
        input_img = keras.layers.Input(shape=x2.shape[1:])
        pair_image_out_list=[]
        input_img_whole_list=[]
        input_img_whole_list.append(input_img_single)
        input_img_multi_list=[]
        for i in range(0,n-1):
            input_img_multi = keras.layers.Input(shape=x2_1.shape[1:])
            input_img_multi_list.append(input_img_multi)
            input_img_whole_list.append(input_img_multi)
            pair_image_out=pair_image_model(input_img_multi)

            pair_image_out_list.append(pair_image_out)
        merged_vector=keras.layers.concatenate(pair_image_out_list[:], axis=-1)#modify this sentence to merge
        merged_model=keras.Model(input_img_multi_list,merged_vector)
        merged_out=merged_model(input_img_multi_list)
        combined_layer = keras.layers.concatenate([single_image_out, merged_out], axis=-1)
        combined_layer = keras.layers.Dropout(0.5)(combined_layer)

        combined = keras.layers.Dense(512, activation='relu')(combined_layer)
        combined = keras.layers.Dropout(0.5)(combined)

        combined = keras.layers.Dense(128, activation='relu')(combined)
        combined = keras.layers.Dropout(0.5)(combined)
        if self.num_classes < 2:
            print('no enough categories')
            sys.exit()
        elif self.num_classes == 2:
            combined=keras.layers.Dense(1, activation='sigmoid')(combined)

            model = keras.Model(input_img_whole_list, combined)
            sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, verbose=0, mode='auto')
        checkpoint1 = ModelCheckpoint(filepath=self.save_dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                      monitor='val_loss',
                                      verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        checkpoint2 = ModelCheckpoint(filepath=self.save_dir + '/weights.hdf5', monitor='val_accuracy', verbose=1,
                                      save_best_only=True, mode='auto', period=1)
        callbacks_list = [checkpoint2, early_stopping]
        self.model = model
        self.callbacks_list = callbacks_list



    def predict_use_model(self, weight_path=""):
        self.load_model_path = weight_path
        test_indel = 0  # the name of data. (0_xdata.npy, 0_ydata.npy, 0_zdata.npy)
        self.update_test_train_data(test_indel, self.epochs)
        self.construct_model(self.x_train)
        model = self.model
        if self.load_model_path is not None:
            model.load_weights(self.load_model_path)

        if not os.path.isdir(self.predict_output_dir):
            os.makedirs(self.predict_output_dir)

        total_y_test_x, total_y_predict_x = [], []
        total_acc_num=0
        total_test_num = 0

        for test_indel in range(self.num_batches):                                      #  for GTRD leave-one-TF-out CV
            self.update_test_train_data(test_indel, self.epochs)
            x_test = self.x_test
            y_test = self.y_test
            z_test = self.z_test
            n = x_test.shape[1]
            x_test_list=[]
            for j in range(0,n):
                x_test_list.append(x_test[:,j,:,:,np.newaxis])

            print('load model and predict')
            y_predict = model.predict(x_test_list)
            flat_predict = y_predict.flatten()
            # acc
            l_acc,l_correct_num = self.cal_acc(flat_predict,y_test)
            total_acc_num += l_correct_num
            total_test_num += len(y_test)
            print('ACC:', l_acc)

            #auc
            y_test_x = [j for j in y_test]
            y_predict_x = [j for j in y_predict]
            for j in y_test:
                total_y_test_x.append(j)
            for j in y_predict:
                total_y_predict_x.append(j)

            fpr, tpr, thresholds = metrics.roc_curve(y_test_x, y_predict_x, pos_label=1)
            l_auc = np.trapz(tpr, fpr)
            print('AUC:', l_auc)

            np.save(self.predict_output_dir + str(test_indel)+'end_y_predict.npy', y_predict)
            np.savetxt(self.predict_output_dir+str(test_indel)+"end_y_predict.csv", y_predict, delimiter=",")

            np.save(self.predict_output_dir +str(test_indel) + 'end_y_test.npy', y_test)
            np.savetxt(self.predict_output_dir +str(test_indel)+ 'end_y_test.csv', y_test, delimiter=",")
            print(z_test)
            df = pd.DataFrame(z_test)
            df.to_csv(self.predict_output_dir +str(test_indel)+ 'end_z_test.csv')

        acc = total_acc_num/total_test_num
        print('Total ACC:', acc)

        fpr, tpr, thresholds = metrics.roc_curve(total_y_test_x, total_y_predict_x, pos_label=1)
        auc = np.trapz(tpr, fpr)
        print('Total AUC:', auc)

    def cal_acc(self,flat_predict,y_test):
        # acc = metrics.accuracy_score(y_predict, y_test)
        batch_acc_num = 0
        for i in range(len(flat_predict)):
            if flat_predict[i] >0.5 and y_test[i]==1 :
                batch_acc_num+=1
            elif flat_predict[i] <0.5 and y_test[i]==0:
                batch_acc_num += 1
        acc = batch_acc_num/len(flat_predict)
        return acc,batch_acc_num


def main_predict():
    #weight_path = "to_predict/mesc_training_weights.hdf5"
    #data_path = "to_predict/boneMarrow_TFdivideNew_topcov10/version11/"

    #predict_output_dir = "to_predict/predict_boneMarrow_withmescWeight/"
    #num_batches = 13
    tcs = use_model_predict(num_batches=args.num_batches,
                                     data_path=args.data_path,
                                     predict_output_dir=args.output_dir)
    tcs.predict_use_model(args.weight_path)


if __name__ == '__main__':
    main_predict()


