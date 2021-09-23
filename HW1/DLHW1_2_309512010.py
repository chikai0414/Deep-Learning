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
def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    
    # select activation function
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
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    return A_curr, memory
def get_cost_value(Y_hat, Y):
    m=Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

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
    
 
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));


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
def train(X, Y, network_arch, epochs, learning_rate):
	params_values = init_layers(network_arch)
	cost_curve=[]
	accuracy_history = []
	cashe_history =[]
	batch_size = 32
	for i in range(epochs):
			m = Y.shape[1]
			x1, y1 = shuffle(X.T, Y.T, random_state=0)  
			num_complete_minibatches = math.floor(m/batch_size)
			cost_sum = 0
			accuracy_sum = 0
			for k in range(0,num_complete_minibatches):
            # 按照batch分別丟進train
					x_batch=x1[int(batch_size)*k:int(batch_size)*(k+1)].T
					y_batch=y1[int(batch_size)*k:int(batch_size)*(k+1)].T
					Y_hat, cashe = full_forward_propagation(x_batch, params_values, network_arch)
					cost = get_cost_value(Y_hat,y_batch)
        
					grads_values = full_back_propagation(Y_hat, y_batch, cashe, params_values, network_arch)
					params_values = update(params_values, grads_values, network_arch, learning_rate)
					cost_sum += cost * batch_size
					accuracy_sum += get_accuracy_value(Y_hat, y_batch) * batch_size
			if Y.shape[1]/batch_size != 0 :
					x_batch=x1[int(batch_size)*k:m].T
					y_batch=y1[int(batch_size)*k:m].T
					Y_hat, cashe = full_forward_propagation(x_batch, params_values, network_arch)
					cost = get_cost_value(Y_hat,y_batch)
        
					grads_values = full_back_propagation(Y_hat, y_batch, cashe, params_values, network_arch)
					params_values = update(params_values, grads_values, network_arch, learning_rate)
					cost_sum += cost * (m % batch_size)
					accuracy_sum += get_accuracy_value(Y_hat, y_batch) * (m % batch_size)
       
					cost_curve.append(cost_sum/m)
					accuracy_history.append(accuracy_sum/m)
			if(i % 10 == 0):
					Y_test_hat, c = full_forward_propagation(np.transpose(x_test), params_values, network_arch)
					acc_t = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
					cashe_history.append(c)
					if(1):
							print("Iter: {:05} | acc: {:.4f} - loss: {:.4f} | acct: {:.4f}".format(i, accuracy_sum/m,cost_sum/m,acc_t))
	return params_values,cost_curve,cashe_history
def update(params_values, grads_values, network_arch, learning_rate):
    for layer_idx, layer in enumerate(network_arch, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values
def get_accuracy_value(Y_hat, Y):
    Y_hat_ = check_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()
def check_class(probs):
    p = np.copy(probs)
    p[p>0.5] = 1
    p[p<=0.5]= 0
    return p
if __name__ == "__main__":

	data = pd.read_csv('ionosphere_data.csv',header=None)
	label=data.iloc[:,-1:]
	data = data.drop(data.columns[-1],axis=1)

	#x_train, x_test, y_train, y_test = train_test_split(data, heating, test_size=0.25, random_state=42)
	lab_enc = pd.Categorical(label.iloc[:,0], categories=label.iloc[:,0].unique(), ordered=True)
	label=lab_enc.codes
	x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
	network_arch = [
    	{"input_shape": 34, "output_shape": 32, "activation": "relu"},
    	{"input_shape": 32, "output_shape": 2, "activation": "relu"},
    	{"input_shape":  2, "output_shape": 1, "activation": "sigmoid"},
	]
	params_values,curve,c = train(np.transpose(x_train), np.array([y_train]), network_arch, 1500, 0.0001)
	Y_test_hat, _ = full_forward_propagation(np.transpose(x_test), params_values, network_arch)
	acc_t = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
	loss_t=get_cost_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
	print("\nTesting set | acc: {:.4f} - loss: {:.4f}".format(acc_t,loss_t))
	plt.plot(curve[:])
	plt.show()

	x=np.array(c[:])
	plt.title("2D Feature Epoch 10") # title
	p = plt.scatter(x[1]["Z2"][:][0],x[1]["Z2"][:][1],c=y_test,s=25)
	clas = ['class 1','class 2']
	plt.xlim(-0.6, 0.1)
	l = plt.legend(handles = p.legend_elements()[0],labels = clas)
	plt.show()
	plt.title("2D Feature Epoch 1500") # title
	p = plt.scatter(x[149]["Z2"][:][0],x[149]["Z2"][:][1],c=y_test,s=25)
	clas = ['class 1','class 2']
	plt.xlim(-0.6, 0.1)
	l = plt.legend(handles = p.legend_elements()[0],labels = clas)
	plt.show()
	print("\n---------------------3D-------------------------\n")
	network_arch = [
    	{"input_shape": 34, "output_shape": 32, "activation": "relu"},
    	{"input_shape": 32, "output_shape": 3, "activation": "relu"},
    	{"input_shape":  3, "output_shape": 1, "activation": "sigmoid"},
	]
	params_values,curve,c = train(np.transpose(x_train), np.array([y_train]), network_arch, 1500, 0.0001)
	Y_test_hat, _ = full_forward_propagation(np.transpose(x_test), params_values, network_arch)
	acc_t = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
	loss_t=get_cost_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
	print("\nTesting set | acc: {:.4f} - loss: {:.4f}".format(acc_t,loss_t))
	x=np.array(c[:])

	fig = plt.figure()
	ax = plt.axes(projection="3d")
	plt.title("3D Feature Epoch 10") # title
	ax.scatter3D(x[1]["Z2"][:][0],x[1]["Z2"][:][1],x[1]["Z2"][:][2],c=y_test);
	plt.show()
	fig = plt.figure()
	ax = plt.axes(projection="3d")
	plt.title("3D Feature Epoch 1300") # title
	ax.scatter3D(x[130]["Z2"][:][0],x[130]["Z2"][:][1],x[130]["Z2"][:][2],c=y_test);
	plt.show()

