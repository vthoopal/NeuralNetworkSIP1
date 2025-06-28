import numpy as np
import pandas as pd
weight1file = pd.read_excel(r"C:\Users\vedth\Desktop\sip\NeuralNetwork1\sip1\w1 (3 inputs - 11 nodes).xlsx")
weights1 = np.transpose(weight1file.to_numpy())
correctfile = pd.read_excel(r"C:\Users\vedth\Desktop\sip\NeuralNetwork1\sip1\cross_data (3 inputs - 2 outputs).xlsx")
correct = correctfile.to_numpy()
#print(correct)
weight2file = pd.read_excel(r"C:\Users\vedth\Desktop\sip\NeuralNetwork1\sip1\w2 (from 11 to 2).xlsx")
weights2 = np.transpose(weight2file.to_numpy())

bias1file = pd.read_excel(r"C:\Users\vedth\Desktop\sip\NeuralNetwork1\sip1\b1 (11 nodes).xlsx")
temp1 = np.transpose(bias1file.to_numpy())


bias2file = pd.read_excel(r"C:\Users\vedth\Desktop\sip\NeuralNetwork1\sip1\b2 (2 output nodes).xlsx")
temp2 = np.transpose(bias2file.to_numpy())

def reformat(matrix):
    bias =[]
    for x in matrix:
        for y in x:
            bias.append((int(y*10000))/10000)
    return bias
bias1 =reformat(temp1)
bias2=reformat(temp2)

def activation(x):
    y = 1/(1+np.exp(-x))
    return y

def activation_derivative(x):
    y = x*(1-x)
    return y

def get_inputs(row):
    x1=correct[row][0]
    x2=correct[row][1]
    x3=correct[row][2]
    input_matrix =[x1,x2,x3]
    return input_matrix

def create_output(matrix, bias):
    solve=[]
    for x in matrix:
        solve.append((int(activation(x)*10000))/10000)
    #print(solve)
    #print(bias)
    
    solve2 = np.add(solve, bias)
   # print(solve2)
    return solve2

def forward(row):
    hidden_value = np.dot(get_inputs(row), weights1)
    hidden_output=(create_output(hidden_value, bias1))

    final_value = np.dot(hidden_output, weights2)
    final_output = create_output(final_value, bias2)
    return final_output

def iterate():
    for i in range(314):
        print(forward(i))
iterate()