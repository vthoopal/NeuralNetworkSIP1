import numpy as np
import pandas as pd
weight1file = pd.read_excel(r"C:\Users\vedth\Desktop\sip\NeuralNetwork1\sip1\w1 (3 inputs - 11 nodes).xlsx")
weights1 = np.transpose(weight1file.to_numpy())

weight2file = pd.read_excel(r"C:\Users\vedth\Desktop\sip\NeuralNetwork1\sip1\w2 (from 11 to 2).xlsx")
weights2 = np.transpose(weight2file.to_numpy())
print(weights2)


def activation(x):
    y = 1/(1+np.exp(-x))
    return y

def activation_derivative(x):
    y = x*(1-x)
    return y

def get_inputs():
    x1=1
    x2=2
    x3=3
    input_matrix =[x1,x2,x3]
    return input_matrix

hidden_value = np.dot(get_inputs(), weights1)
print(hidden_value)

def create_output(matrix):
    solve=[]
    for x in matrix:
        solve.append((int(activation(x)*10000))/10000)
    return solve

hidden_output=(create_output(hidden_value))

final_value = np.dot(hidden_output, weights2)
final_output = create_output(final_value)
