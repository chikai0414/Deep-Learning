import numpy as np 
import pandas as pd
import keras
import random
import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Embedding, \
    LSTM, concatenate, Dense,Bidirectional,Dropout,Input,SimpleRNN
def sample(preds, temp=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    preds = preds.reshape(-1)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

if __name__ == "__main__":
	data_URL = 'shakespeare.txt'
	with io.open( data_URL,'r',encoding='utf8' )as f:
		text=f.read()
	vocab =set(text)
	vocab_to_int = {c:i for i,c in enumerate(vocab)}
	int_to_vocab = dict(enumerate(vocab))
	train_data = np.array( [ vocab_to_int[c] for c in text ] , dtype=np.int32)
	text = " " * 10 + text 
	chars = sorted(list(set(text)))
	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))

	maxlen = 20
	step = 1
	sentences = []
	next_chars = []
	dataX = []
	dataY = []
	for i in range(0, len(text) - maxlen, 1):
	    sentences.append(text[i : i + maxlen])
	    next_chars.append(text[i + maxlen])
	    dataX.append([char_indices[char] for char in text[i : i + maxlen]])
	    dataY.append(char_indices[text[i + maxlen]])

	n_patterns = len(dataX)
	X = np.reshape(dataX, (n_patterns, maxlen,1))
	Y = np_utils.to_categorical(dataY)
	X = X / float(Y.shape[1])
	Y = np_utils.to_categorical(dataY)
	X_train, X_val ,Y_train, Y_val= train_test_split(X,Y,test_size=0.3, random_state = 20, shuffle=True)

	model = Sequential()
	#e = Embedding(64,2,input_length=30)
	#model.add(e)
	model.add(SimpleRNN(125, input_shape=(X_train.shape[1],X_train.shape[2])))
	model.add(Dropout(0.2))

	model.add(Dense(100, activation='relu'))
	model.add(Dense(Y_train.shape[1], activation='softmax'))
	model.summary()
	# load the network weights
	#optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
	model.compile(loss='categorical_crossentropy', optimizer="adam")
	epochs = 10
	batch_size = 128
	train_loss = []
	val_loss = []

	for epoch in range(epochs):
	    history = model.fit(X_train,Y_train, batch_size=batch_size, epochs=1)
	    train_loss.append(history.history['loss'])
	    loss = model.evaluate(X_val, Y_val)
	    val_loss.append(loss)
	    print(loss)
	    print("Generating text after epoch: %d" % (epoch+1))
	    
	    test = []
	    test_data=' '*14+'JULIET'
	    test.append([char_indices[char] for char in test_data[0:maxlen]])
	    pattern = test[0]
	    generated = ""
	    print('Seed: ' + test_data)
	    for i in range(200):
	        x = np.reshape(pattern, (1, len(pattern),1))
	        x = x / float(Y_train.shape[1])
	        prediction = model.predict(x, verbose=0)
	        #index = np.argmax(prediction)
	        index = sample(prediction,0.6)
	        result = indices_char[index]
	        generated+=result
	        pattern.append(index)
	        pattern = pattern[1:len(pattern)]
	    print("Generated: ", generated)
	plt.plot(train_loss,label = "train_loss")
	plt.ylabel("loss")
	plt.xlabel("epoch")
	plt.plot(val_loss,label = "val_loss")
	leg = plt.legend(loc='upper right', shadow=True) 
	plt.show()

	print("Training error rate = ")
	print((train_loss[-1]))
	print("Validation error rate = ")
	print((val_loss[-1]))

	test = []
	generated = ""
	test_data='              JULIET'
	test.append([char_indices[char] for char in test_data[0:20]])
	pattern = test[0]
	i = 0
	while i <= 15:
	    x = np.reshape(pattern, (1, len(pattern),1))
	    x = x / float(Y.shape[1])
	    prediction = model.predict(x, verbose=0)
	    index = np.argmax(prediction)
	    index = sample(prediction,0.4)
	    result = indices_char[index]
	    seq_in = [indices_char[value] for value in pattern]
	    generated+=result
	    pattern.append(index)
	    pattern = pattern[1:len(pattern)]
	    if result =='\n' :
	        i = i + 1
	print("\n\nJULIET" + generated)

	model = Sequential()
	#e = Embedding(64,2,input_length=30)
	#model.add(e)
	model.add(LSTM(250, input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(250))
	model.add(Dropout(0.2))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(Y_train.shape[1], activation='softmax'))
	model.summary()
	# load the network weights
	#optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
	model.compile(loss='categorical_crossentropy', optimizer="adam")
	epochs = 10
	batch_size = 128
	train_loss = []
	val_loss = []

	for epoch in range(epochs):
		history = model.fit(X_train,Y_train, batch_size=batch_size, epochs=1)
		train_loss.append(history.history['loss'])
		loss = model.evaluate(X_val, Y_val)
		val_loss.append(loss)
		print(loss)
		print("Generating text after epoch: %d" % (epoch+1))
		test = []
		test_data=' '*14+'JULIET'
		test.append([char_indices[char] for char in test_data[0:maxlen]])
		pattern = test[0]
		generated = ""
		print('Seed: ' + test_data)
		for i in range(200):
			x = np.reshape(pattern, (1, len(pattern),1))
			x = x / float(Y_train.shape[1])
			prediction = model.predict(x, verbose=0)
	        #index = np.argmax(prediction)
			index = sample(prediction,0.6)
			result = indices_char[index]
			generated+=result
			pattern.append(index)
			pattern = pattern[1:len(pattern)]
		print("Generated: ", generated)
	plt.ylabel("loss")
	plt.xlabel("epoch")
	plt.plot(train_loss,label = "train_loss")
	plt.plot(val_loss,label = "val_loss")
	leg = plt.legend(loc='upper right', shadow=True) 
	plt.show()
	print("Training error rate = ")
	print((train_loss[-1]))
	print("Validation error rate = ")
	print((val_loss[-1]))
	test = []
	generated = ""
	test_data='              JULIET'
	test.append([char_indices[char] for char in test_data[0:20]])
	pattern = test[0]
	i = 0
	while i <= 15:
	    x = np.reshape(pattern, (1, len(pattern),1))
	    x = x / float(Y.shape[1])
	    prediction = model.predict(x, verbose=0)
	    index = np.argmax(prediction)
	    index = sample(prediction,0.4)
	    result = indices_char[index]
	    seq_in = [indices_char[value] for value in pattern]
	    generated+=result
	    pattern.append(index)
	    pattern = pattern[1:len(pattern)]
	    if result =='\n' :
	        i = i + 1
	print("\n\nJULIET" + generated)