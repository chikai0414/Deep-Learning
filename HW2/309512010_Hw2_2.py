import numpy as np
import struct
import matplotlib.pyplot as plt
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from keras import Input
from keras import layers
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.models import Model
from keras.utils import np_utils, plot_model
from keras.regularizers import l2
from keras.models import Model
from keras.datasets import cifar10
from keras.layers import  BatchNormalization

##-----------------------------------------------------------------------
def draw_weight(train_acc,val_acc,test_acc,model):
    weight1, bias1 = model.layers[0].get_weights()
    weight2, bias2 = model.layers[1].get_weights()
    weight3, bias3 = model.layers[5].get_weights()
    weight4, bias4 = model.layers[6].get_weights()
    plt.figure(figsize=(10,20))
    plt.subplot(321)
    plt.title("Accuracy",fontsize=12)
    plt.plot(train_acc,label = "train_acc")
    plt.plot(val_acc,label = "val_acc")
    plt.plot(test_acc,label = "test_acc")
    leg = plt.legend(loc='lower right', shadow=True) 
    plt.subplot(322)
    plt.title("Learning Curve",fontsize=12)
    plt.plot(train_loss,label = "crossentropy")
    leg = plt.legend(loc='upper right', shadow=True)
    plt.subplot(323)
    plt.title("Hist of conv1 : weights",fontsize=12)
    plt.hist(weight1.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "weights")
    plt.legend()
    plt.ylabel('number')
    plt.subplot(324)
    plt.title("Hist of conv1 : biases",fontsize=12)
    plt.hist(bias1.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "bias")
    plt.legend()
    plt.ylabel('number')
    plt.subplot(325)
    plt.title("Hist of conv2 : weights",fontsize=12)
    plt.hist(weight2.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "weights")
    plt.legend()
    plt.ylabel('number')
    plt.subplot(326)
    plt.title("Hist of conv2 : biases",fontsize=12)
    plt.hist(bias2.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "bias")
    plt.legend()
    plt.ylabel('number')
    plt.show()

    plt.figure(figsize=(10,10))
    plt.subplot(221)
    plt.title("Hist of conv3 : weights",fontsize=12)
    plt.hist(weight3.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "bias")
    plt.legend()
    plt.ylabel('number')
    plt.subplot(222)
    plt.title("Hist of conv3 : biases",fontsize=12)
    plt.hist(bias3.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "bias")
    plt.legend()
    plt.ylabel('number')
    plt.subplot(223)
    plt.title("Hist of conv4 : weights",fontsize=12)
    plt.hist(weight4.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "bias")
    plt.legend()
    plt.xlabel('value')
    plt.ylabel('number')
    plt.subplot(224)
    plt.title("Hist of conv4 : biases",fontsize=12)
    plt.hist(bias4.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "bias")
    plt.legend()
    plt.xlabel('value')
    plt.ylabel('number')
    plt.show()

    plt.figure(figsize=(10,20))
    weight1, bias1 = model.layers[10].get_weights()
    weight2, bias2 = model.layers[11].get_weights()
    weight3, bias3 = model.layers[16].get_weights()
    plt.subplot(321)
    plt.title("Hist of conv5 : weights",fontsize=12)
    plt.hist(weight1.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "weights")
    plt.legend()
    #plt.xlabel('value')
    plt.ylabel('number')

    plt.subplot(322)
    plt.title("Hist of conv5 : biases",fontsize=12)
    plt.hist(bias1.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "bias")
    plt.legend()
    #plt.xlabel('value')
    plt.ylabel('number')

    plt.subplot(323)

    plt.title("Hist of conv6 : weights",fontsize=12)
    plt.hist(weight2.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "weights")
    plt.legend()

    plt.ylabel('number')

    plt.subplot(324)
    plt.title("Hist of conv6 : biases",fontsize=12)
    plt.hist(bias2.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "bias")
    plt.legend()
    plt.ylabel('number')

    plt.subplot(325)
    plt.title("Hist of dense1 : weights",fontsize=12)
    plt.hist(weight3.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "bias")
    plt.legend()
    plt.ylabel('number')
    plt.subplot(326)
    plt.title("Hist of dense1 : biases",fontsize=12)
    plt.hist(bias3.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "bias")
    plt.legend()
    plt.ylabel('number')
    plt.show()
def draw_picture(model,test_label):
    count = 0
    index = 0
    predictions = model.predict(x_test)
    for i in range(10000):
        if predict_to_number(predictions[i]) !=predict_to_number(y_test[i]) :
            plt.imshow(x_test[i].reshape(32,32), cmap='gray')
            t = "Label  :"+ str(cifar_name[predict_to_number(y_test[i])])
            l = "Predict:"+ str(cifar_name[predict_to_number(predictions[i])])
            plt.title(t)
            plt.xlabel(l)
            plt.show()
            count+=1
            if count == 2:
                index = i
                break
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(x_test[index].reshape(1,32,32,1))
    draw_activations(activations)

def draw_activations(activations):
    plt.figure(figsize=(6,6))
    plt.subplot(331)
    plt.imshow(activations[0][0, :, :,0], cmap='gray')
    plt.subplot(332)
    plt.imshow(activations[0][0, :, :,2], cmap='gray')
    plt.subplot(333)
    plt.imshow(activations[0][0, :, :,4], cmap='gray')
    plt.subplot(334)
    plt.imshow(activations[5][0, :, :,0], cmap='gray')
    plt.subplot(335)
    plt.imshow(activations[5][0, :, :,2], cmap='gray')
    plt.subplot(336)
    plt.imshow(activations[5][0, :, :,4], cmap='gray')
    plt.subplot(337)
    plt.imshow(activations[11][0, :, :,0], cmap='gray')
    plt.subplot(338)
    plt.imshow(activations[11][0, :, :,2], cmap='gray')
    plt.subplot(339)
    plt.imshow(activations[11][0, :, :,4], cmap='gray')
    plt.show()

def predict_to_number(Z):
    return np.argmax(Z)

if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    X_tr = np.zeros((50000,32,32))
    X_te = np.zeros((10000,32,32))
    x_train = X_train.astype('float32')/255
    x_test = X_test.astype('float32')/255

    for n in range(50000):
        X_tr[n]=np.dot(X_train[n],[0.299,0.587,0.114])
    for n in range(10000):
        X_te[n]=np.dot(X_test[n],[0.299,0.587,0.114])
    X_tr=X_tr.reshape(50000,32,32,1)
    x_test=X_te.reshape(10000,32,32,1)
    mean = np.mean(X_tr,axis=(0,1,2,3))
    std = np.std(X_tr,axis=(0,1,2,3))
    X_tr = (X_tr-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)
    y_train = np_utils.to_categorical(Y_train)
    y_test = np_utils.to_categorical(Y_test)
    x_train,x_val,y_train,y_val=train_test_split(X_tr,y_train,test_size=0.1, random_state=0)


    cifar_name=["airplane",'automobile','bird','cat','deer','dog','frog','horse',"ship","truck"]

    plt.imshow(X_test[1254])
    plt.show()
    plt.imshow(x_test[1254].reshape(32,32), cmap='gray')
    plt.show()

    ep = 50

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3,input_shape=(32,32,1), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=3,activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=3,activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=3,activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=128, kernel_size=3,activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=3,activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    test_acc = []
    train_acc = []
    val_acc = []
    train_loss= []
    for i in range(ep):
            history = model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=1)
            loss, accuracy = model.evaluate(x_test, y_test ,verbose=0)
            test_acc.append(accuracy)
            loss, accuracy = model.evaluate(x_val, y_val,verbose=0)
            val_acc.append(accuracy)
            train_acc.append(history.history['accuracy'])
            train_loss.append(history.history['loss'])

    print("\nKernel size = 3 | Strides = (1,1) | 50 epochs")
    print("\ntrain_acc:",train_acc[-1])
    print("val_acc:",val_acc[-1])
    print("test_acc:",test_acc[-1])
    print("\n----------------------")

    draw_weight(train_acc,val_acc,test_acc,model)
    draw_picture(model,y_test)

#------------------------------------------------------------
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=5,input_shape=(32,32,1), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=5,activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=5,activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=5,activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=128, kernel_size=5,activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=5,activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    test_acc = []
    train_acc = []
    val_acc = []
    train_loss= []
    for i in range(ep):
            history = model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=1)
            loss, accuracy = model.evaluate(x_test, y_test,verbose=0)
            test_acc.append(accuracy)
            loss, accuracy = model.evaluate(x_val, y_val,verbose=0)
            val_acc.append(accuracy)
            train_acc.append(history.history['accuracy'])
            train_loss.append(history.history['loss'])

    print("\nKernel size = 5 | Strides = (1,1) | 50 epochs")
    print("\ntrain_acc:",train_acc[-1])
    print("val_acc:",val_acc[-1])
    print("test_acc:",test_acc[-1])
    print("\n----------------------")

#------------------------------------------------------------
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=7,input_shape=(32,32,1), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=7,activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=7,activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=7,activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=128, kernel_size=7,activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=7,activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    test_acc = []
    train_acc = []
    val_acc = []
    train_loss= []
    for i in range(ep):
            history = model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=1)
            loss, accuracy = model.evaluate(x_test, y_test,verbose=0)
            test_acc.append(accuracy)
            loss, accuracy = model.evaluate(x_val, y_val,verbose=0)
            val_acc.append(accuracy)
            train_acc.append(history.history['accuracy'])
            train_loss.append(history.history['loss'])

    print("\nKernel size = 7 | Strides = (1,1) | 50 epochs")
    print("\ntrain_acc:",train_acc[-1])
    print("val_acc:",val_acc[-1])
    print("test_acc:",test_acc[-1])
    print("\n----------------------")

#----------------------------------------------------------------    
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=5,strides=(2,2),input_shape=(32,32,1), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=5,strides=(2,2),activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=5,activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=5,activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=128, kernel_size=5,activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=5,activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    test_acc = []
    train_acc = []
    val_acc = []
    train_loss= []
    for i in range(ep):
            history = model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=1)
            loss, accuracy = model.evaluate(x_test, y_test,verbose=0)
            test_acc.append(accuracy)
            loss, accuracy = model.evaluate(x_val, y_val,verbose=0)
            val_acc.append(accuracy)
            train_acc.append(history.history['accuracy'])
            train_loss.append(history.history['loss'])

    print("\nKernel size = 5 | Fist two Layer Strides = (2,2) | 50 epochs")
    print("\ntrain_acc:",train_acc[-1])
    print("val_acc:",val_acc[-1])
    print("test_acc:",test_acc[-1])
    print("\n----------------------")

#----------------------------------------------------------------
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), input_shape=(32,32,1), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=3,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=3,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=3,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001),activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=128, kernel_size=3,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=3,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001),activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(10,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    test_acc = []
    train_acc = []
    val_acc = []
    train_loss= []
    for i in range(ep):
        history = model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=1)
        loss, accuracy = model.evaluate(x_test, y_test,verbose=0)
        test_acc.append(accuracy)
        loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
        val_acc.append(accuracy)
        train_acc.append(history.history['accuracy'])
        train_loss.append(history.history['loss'])
    
    print("\nKernel size = 3 | Strides = (1,1) | 50 epochs | Add L2")
    print("\ntrain_acc:",train_acc[-1])
    print("val_acc:",val_acc[-1])
    print("test_acc:",test_acc[-1])
    print("\n----------------------")

    draw_weight(train_acc,val_acc,test_acc,model)


