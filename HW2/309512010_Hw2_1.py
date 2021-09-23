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
train_images_idx3_ubyte_file = 'MNIST_data/train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = 'MNIST_data/train-labels.idx1-ubyte'
test_images_idx3_ubyte_file = 'MNIST_data/t10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = 'MNIST_data/t10k-labels.idx1-ubyte'
##----------------------------read MNIST-------------------------------
def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)

    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header) 
    fmt_image = '>' + str(image_size) + 'B'  
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    #print('magic_number:%d, images: %d' % (magic_number, num_images))

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels
def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)
def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)
def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)
def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


##-----------------------------------------------------------------------
def draw_weight(train_acc,val_acc,test_acc,model):
    weight1, bias1 = model.layers[0].get_weights()
    weight2, bias2 = model.layers[2].get_weights()
    weight3, bias3 = model.layers[4].get_weights()
    weight4, bias4 = model.layers[5].get_weights()
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
    plt.title("Hist of dense1 : weights",fontsize=12)
    plt.hist(weight3.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "bias")
    plt.legend()
    plt.ylabel('number')
    plt.subplot(222)
    plt.title("Hist of dense1 : biases",fontsize=12)
    plt.hist(bias3.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "bias")
    plt.legend()
    plt.ylabel('number')
    plt.subplot(223)
    plt.title("Hist of dense2 : weights",fontsize=12)
    plt.hist(weight4.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "bias")
    plt.legend()
    plt.xlabel('value')
    plt.ylabel('number')
    plt.subplot(224)
    plt.title("Hist of dense2 : biases",fontsize=12)
    plt.hist(bias4.reshape(-1), color = 'gray',bins = 75,cumulative = False, label = "bias")
    plt.legend()
    plt.xlabel('value')
    plt.ylabel('number')
    plt.show()

def draw_picture(model,test_label):
    count = 0
    index = 0
    predictions = model.predict(test_images)
    for i in range(10000):
        if predict_to_number(predictions[i]) != test_label[i] :
            plt.imshow(test_images[i].reshape(28,28), cmap='gray')
            t = "Label  :"+ str(predict_to_number(test_labels[i]))
            l = "Predict:"+ str(predict_to_number(predictions[i]))
            plt.title(t)
            plt.xlabel(l)
            plt.show()
            count+=1
            if count == 2:
                index = i
                break
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(test_images[index].reshape(1,28,28,1))
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
    plt.imshow(activations[1][0, :, :,0], cmap='gray')
    plt.subplot(335)
    plt.imshow(activations[1][0, :, :,2], cmap='gray')
    plt.subplot(336)
    plt.imshow(activations[1][0, :, :,4], cmap='gray')
    plt.subplot(337)
    plt.imshow(activations[2][0, :, :,0], cmap='gray')
    plt.subplot(338)
    plt.imshow(activations[2][0, :, :,2], cmap='gray')
    plt.subplot(339)
    plt.imshow(activations[2][0, :, :,4], cmap='gray')
    plt.show()

def predict_to_number(Z):
    return np.argmax(Z)
if __name__ == '__main__':
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()

    train_label = train_labels
    test_label = test_labels
    train_labels = np_utils.to_categorical(train_labels)
    test_labels = np_utils.to_categorical(test_labels)
    train_images = train_images.reshape(60000,28,28,1)
    train_images,val_images,train_labels, val_labels=train_test_split(train_images,train_labels,test_size=5000, random_state=0)
    test_images = test_images.reshape(10000,28,28,1)


    ep = 50

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3,strides=(1,1), input_shape=(28,28,1), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=32, kernel_size=3,strides=(1,1),activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    test_acc = []
    train_acc = []
    val_acc = []
    train_loss= []
    for i in range(ep):
        history = model.fit(train_images, train_labels, epochs=1, batch_size=64, verbose=1)
        loss, accuracy = model.evaluate(test_images, test_labels,verbose=0)
        test_acc.append(accuracy)
        loss, accuracy = model.evaluate(val_images, val_labels, verbose=0)
        val_acc.append(accuracy)
        train_acc.append(history.history['accuracy'])
        train_loss.append(history.history['loss'])

    print("\nKernel size = 3 | Strides = (1,1) | 50 epochs")
    print("\ntrain_acc:",train_acc[-1])
    print("val_acc:",val_acc[-1])
    print("test_acc:",test_acc[-1])
    print("\n----------------------")

    draw_weight(train_acc,val_acc,test_acc,model)
    draw_picture(model,test_label)
##-------------------------------------------------------------------

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=5,strides=(1,1), input_shape=(28,28,1), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=32, kernel_size=5,strides=(1,1),activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    test_acc = []
    train_acc = []
    val_acc = []
    train_loss= []
    for i in range(ep):
        history = model.fit(train_images, train_labels, epochs=1, batch_size=64, verbose=1)
        loss, accuracy = model.evaluate(test_images, test_labels,verbose=0)
        test_acc.append(accuracy)
        loss, accuracy = model.evaluate(val_images, val_labels, verbose=0)
        val_acc.append(accuracy)
        train_acc.append(history.history['accuracy'])
        train_loss.append(history.history['loss'])

    print("\nKernel size = 5 | Strides = (1,1) | 50 epochs")
    print("\ntrain_acc:",train_acc[-1])
    print("val_acc:",val_acc[-1])
    print("test_acc:",test_acc[-1])
    print("\n----------------------")
##-------------------------------------------------------------------

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=7,strides=(1,1), input_shape=(28,28,1), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=32, kernel_size=7,strides=(1,1),activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    test_acc = []
    train_acc = []
    val_acc = []
    train_loss= []
    for i in range(ep):
        history = model.fit(train_images, train_labels, epochs=1, batch_size=64, verbose=1)
        loss, accuracy = model.evaluate(test_images, test_labels,verbose=0)
        test_acc.append(accuracy)
        loss, accuracy = model.evaluate(val_images, val_labels, verbose=0)
        val_acc.append(accuracy)
        train_acc.append(history.history['accuracy'])
        train_loss.append(history.history['loss'])

    print("\nKernel size = 7 | Strides = (1,1) | 50 epochs")
    print("\ntrain_acc:",train_acc[-1])
    print("val_acc:",val_acc[-1])
    print("test_acc:",test_acc[-1])
    print("\n----------------------")
##------------------------------------------------------------------------

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=5,strides=(2,2), input_shape=(28,28,1), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=32, kernel_size=5,strides=(2,2),activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    test_acc = []
    train_acc = []
    val_acc = []
    train_loss= []
    for i in range(ep):
        history = model.fit(train_images, train_labels, epochs=1, batch_size=64, verbose=1)
        loss, accuracy = model.evaluate(test_images, test_labels,verbose=0)
        test_acc.append(accuracy)
        loss, accuracy = model.evaluate(val_images, val_labels, verbose=0)
        val_acc.append(accuracy)
        train_acc.append(history.history['accuracy'])
        train_loss.append(history.history['loss'])

    print("\nKernel size = 5 | Strides = (2,2) | 50 epochs")
    print("\ntrain_acc:",train_acc[-1])
    print("val_acc:",val_acc[-1])
    print("test_acc:",test_acc[-1])
    print("\n----------------------")
##------------------------------------------------------------------------

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, input_shape=(28,28,1), kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001),activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=32, kernel_size=3, kernel_regularizer=l2(0.001),activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(256,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), activation='relu'))
    model.add(Dense(10, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001),activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    test_acc = []
    train_acc = []
    val_acc = []
    train_loss= []
    for i in range(ep):
        history = model.fit(train_images, train_labels, epochs=1, batch_size=64, verbose=1)
        loss, accuracy = model.evaluate(test_images, test_labels,verbose=0)
        test_acc.append(accuracy)
        loss, accuracy = model.evaluate(val_images, val_labels, verbose=0)
        val_acc.append(accuracy)
        train_acc.append(history.history['accuracy'])
        train_loss.append(history.history['loss'])
    print("\nKernel size = 3 | Strides = (1,1) | 50 epochs | Add L2")

    print("\ntrain_acc:",train_acc[-1])
    print("val_acc:",val_acc[-1])
    print("test_acc:",test_acc[-1])
    print("\n----------------------")
    draw_weight(train_acc,val_acc,test_acc,model)


