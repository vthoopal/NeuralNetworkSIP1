import numpy as np
import pandas as pd
weight1file = pd.read_excel(r"C:\Users\vedth\Desktop\sip\NeuralNetwork1\sip1\w1 (3 inputs - 11 nodes).xlsx")#input to node weight
weights1 = np.transpose(weight1file.to_numpy())
correctfile = pd.read_excel(r"C:\Users\vedth\Desktop\sip\NeuralNetwork1\sip1\cross_data (3 inputs - 2 outputs).xlsx")#
correct = correctfile.to_numpy()
weight2file = pd.read_excel(r"C:\Users\vedth\Desktop\sip\NeuralNetwork1\sip1\w2 (from 11 to 2).xlsx")
weights2 = np.transpose(weight2file.to_numpy())
learningRate = 0.7
bias1file = pd.read_excel(r"C:\Users\vedth\Desktop\sip\NeuralNetwork1\sip1\b1 (11 nodes).xlsx")
temp1 = np.transpose(bias1file.to_numpy())

bias2file = pd.read_excel(r"C:\Users\vedth\Desktop\sip\NeuralNetwork1\sip1\b2 (2 output nodes).xlsx")
temp2 = np.transpose(bias2file.to_numpy())

def reformat(matrix):#fixes the matrix
    bias =[]
    for x in matrix:
        for y in x:
            bias.append(y)
    return bias
bias1 =reformat(temp1)
bias2=reformat(temp2)

def activation(x):
    y = 1/(1+np.exp(-x))
    return y

def activation_derivative(x):
    y = x*(1-x)
    return y

def get_correct(row):#collects the correct output from the file
    list=[]
    C1 = correct[row][3]
    C2= correct[row][4]
    list=[C1,C2]
    return list

def get_inputs(row):#collects the inputs
    x1=correct[row][0]
    x2=correct[row][1]
    x3=correct[row][2]
    input_matrix =[x1,x2,x3]
    return input_matrix

def activeDerivative_matrix(values):
    x = len(values)
    matrix = np.zeros((x,x))
    for i in values:
        matrix[values.index(i)][values.index(i)] = activation_derivative(i)  
    return matrix

def calculate_outputError(true, product, sumsquare):
    errorlist =[]
    z =np.subtract(true,product)
    y=activeDerivative_matrix( product)
    errorlist = np.dot(z,y)
    for i in errorlist:
        sumsquare = sumsquare+(i**2)
    return errorlist,sumsquare

def calculate_hiddenError(finError, outputs):
    x =np.transpose(finError)
    Weight_sum = np.dot(weights2, x)
    y = activeDerivative_matrix(outputs)
    
    hidList=[]
    hidList= np.dot(Weight_sum,y)
    return(hidList) 

def create_output(matrix, bias):
    solve=[]
    matrix = np.add(matrix, bias) #Adds bias
    for x in matrix:
        solve.append(activation(x)) #runs the activation function for each value
    return solve

def forward(row):
    hidden_value = np.dot(get_inputs(row), weights1) #dot product of input and weight (not including bias)
    hidden_output=(create_output(hidden_value, bias1))

    final_value = np.dot(hidden_output, weights2)
    final_output = create_output(final_value, bias2)
    return hidden_output, final_output

def updateOutput(error, hiddenOutput, bias, weight):
        x =learningRate*error
        bias = np.add(bias,x)
        hiddenOutput =np.array([hiddenOutput])
        y = np.transpose(hiddenOutput)
        error =np.array([error])
        z = learningRate*(np.dot(y,error))
        weight = np.add(weight,z)
        return bias,weight

def updateHidden(error, row, bias, weight):
    input=[]
    for i in range(3):
        input.append(correct[row][i])
    x = learningRate*error
    input = np.array([input])
    error = np.array([error])
    bias = np.add(bias,x)
    y = np.transpose(input)
    z = learningRate*(np.dot(y,error))
    weight = np.add(weight,z)
    return bias,weight


def main(b1,b2,w1,w2):
    for n in range(1): #Option for multiple epochs
        SSE =0
        for i in range(314):#iterates over every set
            layer1, layer2= forward(i)
            answer = get_correct(i)
            errorO, SSE = calculate_outputError(answer, layer2, SSE)
            errorH = calculate_hiddenError(errorO, layer1)
            b2, w2 = updateOutput(errorO, layer1, b2, w2)
            b1, w1 = updateHidden(errorH, i, b1, w1)
           # print(errorO)
           # print(errorH)
        n = n +1
    SSE = 0 #calculate sse only after weights have been updated
    for i in range(314):
        layer1, layer2= forward(i)
        answer = get_correct(i)
        errorO, SSE = calculate_outputError(answer, layer2, SSE)
        errorH = calculate_hiddenError(errorO, layer1)
    print(SSE)
main(bias1,bias2,weights1,weights2)