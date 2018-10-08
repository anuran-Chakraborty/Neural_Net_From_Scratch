import numpy as np
from math import exp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# This program creates a neural network and implements backpropagation
"""
Data structures used in this program
There are a total of L layers
y : (ONE DIMENSIONAL ARRAY) desired value of node j in an output layer L
	y[j] denotes the value of node j in output layer L fro single training example

C0: (FLOAT) loss function for a single training example

w :	(THREE DIMENSIONAL ARRAY) denotes the weights of each connection of each layer
	w[l][j][k] weight of connection that connects node k in layer l-1 to node j in layer l

bias : (TWO DIMENSIONAL ARRAY) denotes bias of node j
	b[l][j] refers to bias of node j in layer l

wj: (TWO DIMENSIONAL ARRAY) denotes weights connected to node j
	w[l][j]  

z : (TWO DIMENSIONAL ARRAY) input for each node in each layer
	z[l][j] denote input for node j in layer l

g : (ONE DIMENSIONAL ARRAY) activation functions for each layer
	g[l] denotes activation function for layer l

a : (TWO DIMENSIONAL ARRAY) activation output of nodes in each layer 
	a[l][j] activation output of node j in layer l

deltaw: (THREE DIMENSIONAL ARRAY) deltaw[l][j][k] denote derivative of cost wrt w[l][j][k]

deltab: (THREE DIMENSIONAL ARRAY) deltab[l][j] denote derivative of cost wrt bias[l][j]

"""

# Class to implement a neural network
class Neural_network:

	# Constructor to initialise the data members
	def __init__(self,n_inputs,n_hidden,nodes_in_hidden,n_outputs):
		"""Constructor
		
		Args:
			n_inputs (INT): Number of inputs
			n_hidden (INT): Number of hidden layer
			nodes_in_hidden (LIST): Number of nodes in each hidden layer
			n_outputs (INT): Number of outputs
			n_epochs (INT): Number of epochs
		"""
		self.n_inputs=n_inputs
		self.n_hidden=n_hidden
		self.nodes_in_hidden=nodes_in_hidden
		self.n_outputs=n_outputs

		# Contains number of nodes in each layer
		self.nodes_in_layers=[n_inputs]+nodes_in_hidden+[n_outputs]
		print(self.nodes_in_layers)
		self.layers=len(self.nodes_in_layers)

		# **********************************************
		# intialise the weights 3d array w. This is implemented as
		# list of numpy 2d arrays
		self.w=list()
		# For the first set
		arr=np.random.rand(n_inputs,nodes_in_hidden[0])
		self.w.append(arr)

		for i in range(self.layers-1):
			arr=np.random.rand(self.nodes_in_layers[i+1],self.nodes_in_layers[i])
			self.w.append(arr)

		# **********************************************

		# **********************************************
		# intialise the delta 3d array deltaw. This is implemented as
		# list of numpy 2d arrays where each 2d array is a layer
		# layer0 is the  input layer
		self.deltaw=list()
		# For the first set
		arr=np.random.rand(n_inputs,nodes_in_hidden[0])
		self.deltaw.append(arr)

		for i in range(self.layers-1):
			arr=np.zeros((self.nodes_in_layers[i+1],self.nodes_in_layers[i]))
			self.deltaw.append(arr)

		# **********************************************

		# ***************** initialise bias ***********************
		self.bias=list()

		for i in range(self.layers):
			arr=np.random.rand(self.nodes_in_layers[i])
			self.bias.append(arr)

		# ************************************************************

		# ******************** initialise deltab **************************
		self.deltab=list()

		for i in range(self.layers):
			arr=np.zeros(self.nodes_in_layers[i])
			self.deltab.append(arr)

		# *************************************************************

		# ******************** initialise deltab **************************
		self.deltact=list()

		for i in range(self.layers):
			arr=np.zeros(self.nodes_in_layers[i])
			self.deltact.append(arr)

		# *************************************************************
		
		# ******************** initialise z **************************
		self.z=list()

		for i in range(self.layers):
			arr=np.zeros(self.nodes_in_layers[i])
			self.z.append(arr)

		# *************************************************************

		# ******************** initialise a **************************
		self.a=list()

		for i in range(self.layers):
			arr=np.zeros(self.nodes_in_layers[i])
			self.a.append(arr)

		# *************************************************************

		# ********************** initialise y *************************
		self.y=np.zeros(n_outputs)
		# *************************************************************

	# Define the sigmoid function for use later
	def sigmoid_function(self,act):
		return 1.0/(1.0+exp(-act))

	# Define the derivative of sigmoid function for use later
	def deriv_sigmoid_function(self,act):
		return self.sigmoid_function(act)*(1.0-self.sigmoid_function(act))	

	# Function to calculate z
	def calc_z(self,inputs,weights,j,l):

		self.z[l][j]=self.bias[l][j]
		for i in range(len(weights)):
			self.z[l][j]+=weights[i]*inputs[i]
		self.a[l][j]=self.sigmoid_function(self.z[l][j])

	def forward_prop(self,row_of_data):
		"""Function to implement forward propagation
		
		Args:
			row_of_data (ARRAY): row_of_data

		First initialise z[0] to row_of_data
		Next forward propagate accordingly

		"""
		# First layer
		self.z[0]=row_of_data
		self.a[0]=row_of_data
		# For remaining layers
		for l in range(1,self.layers):
			for j in range(self.nodes_in_layers[l]):
				self.calc_z(inputs=self.a[l-1],weights=self.w[l][j],j=j,l=l)

	def backprop(self,expected):
		"""Function to implement backpropagation
		
		Args:
			expected (INT): Expected value of output for a given training example
		"""

		# ******* For the output layer gradient calculation is easy **************
		L=self.layers-1
		self.y[expected]=1
		for j in range(self.n_outputs):
			self.deltact[L][j]= 2*(self.a[L][j]-self.y[j])
			for k in range(self.nodes_in_layers[L-1]):
				self.deltaw[L][j][k]=self.deltact[L][j]*(self.deriv_sigmoid_function(self.z[L][j]))*(self.a[L-1][k])

		# Derivative wrt bias
		for j in range(self.n_outputs):
			self.deltab[L][j]=2*(self.a[L][j]-self.y[j])*(self.deriv_sigmoid_function(self.z[L][j]))

		# ************************************************************************

		# *********** For the remaining layers it is a kind of recursive process ***********
		for l in reversed(range(1,L)):
			for j in range(self.nodes_in_layers[l]):
				s=0.0
				# Calculating sum for next layers
				for m in range(self.nodes_in_layers[l+1]):
					s+=(self.w[l+1][m][j])*self.deltact[l+1][m]*self.deriv_sigmoid_function(self.z[l+1][m])
				self.deltact[l][j]=s
				error=self.deriv_sigmoid_function(self.z[l][j])*s

				for k in range(self.nodes_in_layers[l-1]):
					terror=error*self.a[l-1][k]
					self.deltaw[l][j][k]=terror

		# ************************************************************************
		
		# ********************* Derivative wrt to biases ***************************
		for l in reversed(range(1,L)):
			for j in range(self.nodes_in_layers[l]):
				error=self.deriv_sigmoid_function(self.z[l][j])*self.deltact[l][j]
				self.deltab[l][j]=error
		# **************************************************************************


	# Function for updating weights

	def update_weights(self,l_rate):

		for l in range(1,self.layers):
			
			for j in range(self.nodes_in_layers[l]):
				for k in range(self.nodes_in_layers[l-1]):
					self.w[l][j][k]-=l_rate*self.deltaw[l][j][k]

	# Function for updating bias

	def update_bias(self,l_rate):

		for l in range(1,self.layers):
			for j in range(self.nodes_in_layers[l]):
				self.bias[l][j]-=l_rate*self.deltab[l][j]


	def training(self,trainX,trainy,l_rate,n_epochs):
		"""Function to train 
		
		Args:
			trainX (ARRAY): Training features
			trainy (ARRAY): Training labels
			l_rate (FLOAT): Learning rate
			n_epochs (INT): Number of epochs
		"""
		L=self.layers-1

		for epochs in range(n_epochs):
			sum_error=0.0
			for i in range(len(trainX)):
				rowX=trainX[i]
				rowy=trainy[i]
				self.y=np.zeros(self.n_outputs)
				self.y[rowy]=1

				self.forward_prop(rowX)

				#Calcuate error
				sum_error += sum([(self.y[i]-self.a[L][i])**2 for i in range(self.n_outputs)])


				self.backprop(rowy)
				self.update_weights(l_rate)
				self.update_bias(l_rate)

			print('>epoch=%d, lrate=%.3f, error=%.5f' % (epochs, l_rate,sum_error))
		print('Training complete!!')

	def predict(self,row_of_data):
		self.forward_prop(row_of_data)
		return np.argmax(self.a[self.layers-1])

	def prediction_all(self,inputs):
		outputs=list()
		for row in inputs:
			outputs.append(self.predict(row))
		return outputs

def get_accuracy(prediction,actual):
	# Function to get accuracy
	count=0
	for i in range(len(prediction)):
		if prediction[i]==actual[i]:
			count+=1

	return (1.0*count)/len(prediction)

def get_dataset(path):

	df=pd.read_csv(path,header=None)
	print(df.head())
	y =df.iloc[:,len(df.columns)-1]
	X =df.drop(len(df.columns)-1,axis=1)
	X_train, X_test, y_train,  y_test= train_test_split(X,y,train_size=0.5)

	scaler = MinMaxScaler()
	X_train = scaler.fit_transform(X_train)
	X_test  = scaler.fit_transform(X_test)

	y_train=y_train.values
	y_test=y_test.values

	return X_train, X_test, y_train, y_test



# Run the netowrk on data
dataset_name='../wheat-seeds.csv'
l_rate=0.3
n_epochs=500
n_output=3
n_input=7
n_hidden=2
nodes_in_hidden=[5,5]

X_train, X_test, y_train, y_test=get_dataset(dataset_name)
print(type(y_train))
network=Neural_network(n_inputs=n_input, n_hidden=n_hidden, nodes_in_hidden=nodes_in_hidden, n_outputs=n_output)
network.training(trainX=X_train,trainy=y_train,l_rate=l_rate,n_epochs=n_epochs)
pred=network.prediction_all(X_test)
print('Accuracy:'+str(get_accuracy(pred,y_test)))