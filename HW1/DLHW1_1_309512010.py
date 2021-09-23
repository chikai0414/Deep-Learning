import numpy as np 
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from mpl_toolkits import mplot3d

network_arch = [
    {"input_shape": 16, "output_shape": 32, "activation": "relu"},
    {"input_shape": 32, "output_shape": 16, "activation": "relu"},
    {"input_shape": 16, "output_shape": 8, "activation": "relu"},
    {"input_shape": 8, "output_shape": 1, "activation": "linear"},
]
def PCA(data,x):
    Mat = np.array(data[:], dtype='float64')
    p,n = np.shape(Mat) # shape of Mat 
    t = np.mean(Mat, 0) 
    #把整體資料去做平移
    for i in range(p):
        for j in range(n):
             Mat[i,j] = float(Mat[i,j]-t[j])

    #做出cov矩陣
    cov_Mat = np.dot(Mat.T, Mat)/(p-1)
    #讀取eigh value和讀取eighvector 
    U,V = np.linalg.eigh(cov_Mat) 
    U = U[::-1]
    #rereange
    for i in range(n):
        V[i,:] = V[i,:][::-1]


    comp = x
    v = V[:,:comp]
     # data transformation
    data_PCA = np.dot(Mat, v)
    return data_PCA

def init_layers(n):
    np.random.seed(0)
    number_of_layers = len(n)
    pa = {}
    for i, layer in enumerate(n):
        layer_idx = i + 1
        
        input_size = layer["input_shape"]
        output_size = layer["output_shape"]
        
        pa['W' + str(layer_idx)] = np.random.randn(
            output_size, input_size)*0.1
        pa['b' + str(layer_idx)] = np.random.randn(
            output_size, 1)*0.1       
    return pa

def sig(x):    
    return 1 / (1 + np.exp(-x))
def relu(Z):
    return np.maximum(0,Z) 
def linear_back(dA, Z):
    return dA
def sig_back(dA, Z):
    s = sig(Z)
    return dA*s*(1-s)
def linear(Z):
        return Z
def relu_back(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;
def single_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    
    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sig
    elif activation is "linear":
        activation_func = linear
    else:
        raise Exception('Non-supported activation function')
        
    return activation_func(Z_curr), Z_curr
def full_forward_propagation(X, params_values, network_arch):
    memory = {}
    A_curr = X
    
    for idx, layer in enumerate(network_arch):
        layer_idx = idx + 1
        A_prev = A_curr
        
        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        A_curr, Z_curr = single_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    return A_curr, memory
def get_cost_value(Y_hat, Y):
    return np.power(Y_hat-Y,2).sum()
def single_back_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1]
    
    # select activation function
    if activation is "relu":
        back_activation_func = relu_back
    elif activation is "sigmoid":
        back_activation_func = sig_back
    elif activation is "linear":
        back_activation_func = linear_back
    
    dZ_curr = back_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) 
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) 
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr

def full_back_propagation(Y_hat, Y, memory, params_values, n):
        
    grads = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)
    dA_prev = 2 * (Y_hat-Y)

    for layer_idx_prev, layer in reversed(list(enumerate(n))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_back_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads["dW" + str(layer_idx_curr)] = dW_curr
        grads["db" + str(layer_idx_curr)] = db_curr
    
    return grads
def train(X, Y, n, epochs, learning_rate):
	params_values = init_layers(n)
	cost_curve=[]
	accuracy_history = []
	batch_size = 16
	for i in range(epochs):
		x1, y1 = shuffle(X.T, Y.T, random_state=0)  
		num_complete_minibatches = math.floor(576/batch_size)
		cost_history = 0
		for k in range(0,num_complete_minibatches):
            # batch成不同大小丟入train
				x_batch=x1[int(batch_size)*k:int(batch_size)*(k+1)].T
				y_batch=y1[int(batch_size)*k:int(batch_size)*(k+1)].T
				Y_hat, cashe = full_forward_propagation(x_batch, params_values, n)
				cost = get_cost_value(Y_hat,y_batch)
				grads_values = full_back_propagation(Y_hat, y_batch, cashe, params_values, n)
				params_values = update(params_values, grads_values, n, learning_rate)
				cost_history+=cost
		cost_curve.append(cost_history)
		if(i % 10 == 0):
				if(1):
					print("Iter: {:05} | cost: {:.4f} ".format(i,cost_history))
	return params_values,cost_curve
	
def update(params_values, grads_values, network_arch, learning_rate):
    for layer_idx, layer in enumerate(network_arch, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values
def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_
if __name__ == "__main__":
    
	data = pd.read_csv('energy_efficiency_data.csv')
	heating = data.pop('Heating Load')
	cooling = data.pop('Cooling Load')
	# for one hot encoding
	Orient = pd.get_dummies(data['Orientation'])
	Glazing = pd.get_dummies(data['Glazing Area Distribution'])
	data = pd.concat([data,pd.get_dummies(data['Orientation'], prefix='Orientation')],axis=1)
	data = pd.concat([data,pd.get_dummies(data['Glazing Area Distribution'], prefix='Glazing Area Distribution')],axis=1)
	data.drop(['Orientation'],axis=1, inplace=True)
	data.drop(['Glazing Area Distribution'],axis=1, inplace=True)
	x_train = data[:576]
	x_test = data[576:]
	y_train = heating[:576]
	y_test = heating[576:]
	params_values,curve = train(np.transpose(x_train), np.array([y_train]), network_arch, 1000, 0.000001)
	Y_test_hat,_  = full_forward_propagation(np.transpose(x_test), params_values, network_arch)
	Y_train_hat,_ = full_forward_propagation(np.transpose(x_train), params_values, network_arch)
	print("\n-------------------\n")
	print("Learning_rate = 0.000001")
	print("Training RMS error = {:.4f}".format(math.sqrt(get_cost_value(np.array(Y_train_hat),np.array([y_train]))/576)))
	print("Test RMS error = {:.4f}".format(math.sqrt(get_cost_value(np.array(Y_test_hat),np.array([y_test]))/192)))
	print("\n-------------------\n")
#---------------------------------------------------------
	
	plt.title("Prediction for training data") # title
	plt.xlabel("#th case") 
	plt.ylabel("Heating Load") 
	plt.plot(np.array([y_train]).T)
	plt.plot(Y_train_hat.T)
	plt.legend(['Train', 'Predict'])
	plt.show()

#---------------------------------------------------------
	plt.title("Prediction for test data") # title
	plt.xlabel("#th case") 
	plt.ylabel("Heating Load")
	plt.plot(np.array([y_test]).T)
	plt.plot(Y_test_hat.T)
	plt.legend(['Test', 'Predict'])
	plt.show()

#-------------------PCA---------------------------

	plt.title("training curve") # title
	plt.xlabel("Epoch") # y label
	plt.ylabel("Loss") # x label
	plt.plot(curve[:])
	plt.show()

	data_PCA=PCA(data,8)
	x_train = data_PCA[:576]
	x_test = data_PCA[576:]
	y_train = heating[:576]
	y_test = heating[576:]
	network_arch = [
    {"input_shape": 8, "output_shape": 32, "activation": "relu"},
    {"input_shape": 32, "output_shape": 16, "activation": "relu"},
    {"input_shape": 16, "output_shape": 8, "activation": "relu"},
    {"input_shape": 8, "output_shape": 1, "activation": "linear"},
	]
	params_values,curve1 = train(np.transpose(x_train), np.array([y_train]), network_arch, 1000, 0.000001)
	Y_test_hat,_ = full_forward_propagation(np.transpose(x_test), params_values, network_arch)
	plt.title("training curve") # title
	plt.xlabel("Epoch") # y label
	plt.ylabel("Loss") # x label
	plt.ylim(0, 120000)
	plt.plot(curve[:])
	plt.plot(curve1[:])
	plt.legend(['with out PCA', 'PCA n=8'])
	plt.show()
	print("\n-----PCA n = 8-----\n")
	print("Training RMS error = {:.4f}".format(math.sqrt(get_cost_value(np.array(Y_train_hat),np.array([y_train]))/576)))
	print("Test RMS error = {:.4f}".format(math.sqrt(get_cost_value(np.array(Y_test_hat),np.array([y_test]))/192)))
	print("\n-------------------\n")
#---------------------------------------------------------
	
	plt.title("Prediction for training data") # title
	plt.xlabel("#th case") 
	plt.ylabel("Heating Load") 
	plt.plot(np.array([y_train]).T)
	plt.plot(Y_train_hat.T)
	plt.legend(['Train', 'Predict PCA n = 8'])
	plt.show()

#---------------------------------------------------------
	plt.title("Prediction for test data") # title
	plt.xlabel("#th case") 
	plt.ylabel("Heating Load")
	plt.plot(np.array([y_test]).T)
	plt.plot(Y_test_hat.T)
	plt.legend(['Test', 'Predict PCA n = 8'])
	plt.show()

#---------------------------------------------------------


	data_PCA=PCA(data,12)
	x_train = data_PCA[:576]
	x_test = data_PCA[576:]
	y_train = heating[:576]
	y_test = heating[576:]
	network_arch = [
    {"input_shape": 12, "output_shape": 32, "activation": "relu"},
    {"input_shape": 32, "output_shape": 16, "activation": "relu"},
    {"input_shape": 16, "output_shape": 8, "activation": "relu"},
    {"input_shape": 8, "output_shape": 1, "activation": "linear"},
	]
	params_values,curve1 = train(np.transpose(x_train), np.array([y_train]), network_arch, 1000, 0.000001)
	Y_test_hat,_ = full_forward_propagation(np.transpose(x_test), params_values, network_arch)
	plt.title("training curve") # title
	plt.xlabel("Epoch") # y label
	plt.ylabel("Loss") # x label
	plt.ylim(0, 120000)
	plt.plot(curve[:])
	plt.plot(curve1[:])
	plt.legend(['with out PCA', 'PCA n = 12'])
	plt.show()
	print("\n---PCA n = 12------\n")
	print("Training RMS error = {:.4f}".format(math.sqrt(get_cost_value(np.array(Y_train_hat),np.array([y_train]))/576)))
	print("Test RMS error = {:.4f}".format(math.sqrt(get_cost_value(np.array(Y_test_hat),np.array([y_test]))/192)))
	print("\n-------------------\n")

#---------------------------------------------------------
	
	plt.title("Prediction for training data") # title
	plt.xlabel("#th case") 
	plt.ylabel("Heating Load") 
	plt.plot(np.array([y_train]).T)
	plt.plot(Y_train_hat.T)
	plt.legend(['Train', 'Predict PCA n = 12'])
	plt.show()

#---------------------------------------------------------
	plt.title("Prediction for test data") # title
	plt.xlabel("#th case") 
	plt.ylabel("Heating Load")
	plt.plot(np.array([y_test]).T)
	plt.plot(Y_test_hat.T)
	plt.legend(['Test', 'Predict PCA n = 12'])
	plt.show()