import numpy as np
import json, sys

arg = sys.argv
input_file = arg[1]
hyperparam_file = arg[2]


# read data from input file
A = np.loadtxt(input_file)
x = A[:,:-1]
x = np.vstack([np.ones(len(x)),x.T]).T # pad first column with ones
y = A[:,-1]

# read hyperparameters from json file
with open(hyperparam_file, 'r') as f:
	my_dict = json.load(f)
learning_rate = my_dict['learning rate']
num_iter = my_dict['num iter']

# analytic least squares solution
w_analytic = np.linalg.lstsq(x,y)[0]

# Stochastic gradient descent solution
num_data_pts = np.shape(x)[0]
w_gd = np.ones(len(x[0])) # initialize weights
for i in range(num_iter):
# Get random data point from dataset
	rand_int = np.random.randint(0,num_data_pts)
	x_i = x[rand_int]
	y_i = y[rand_int]
		
	# Compute h then update w
	h = np.dot(w_gd,x_i)
	for j in range(len(x_i)):
		w_gd[j] -= learning_rate*(h-y_i)*x_i[j] # gradient descent update rule

# write to output file
temp = input_file.split('/')[-1]
filename = temp.split('.')[0]
filepath = 'data/' + filename + '.out'
f = open(filepath, 'w')
for num in w_analytic:
	string = '{:.4f}'.format(num)
	f.write(string + '\n')
f.write('\n')
for num in w_gd:
	string = '{:.4f}'.format(num)
	f.write(string + '\n')
f.close()
