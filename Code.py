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

def calculate_outputError(true, product, sumsquare):
    errorlist =[]
    for i in true:
       x = (i - product[true.index(i)])
       sumsquare = sumsquare +  x**2 #calculates the sum of squared error
       e = x * activation_derivative(product[true.index(i)]) #Output error = (correct answer - output)*(the activation derivative of output)
       errorlist.append(e)
    return errorlist, sumsquare

def calculate_hiddenError(finError, outputs):
    hidList=[]
    for y in outputs: 
        w=0
        for i in finError:
            w = weights2[outputs.index(y)][finError.index(i)]*i + w #Hidden error = sum(Weights*corresponding output error)*activation derivative of hidden output
        hidError = w*activation_derivative(y)
        hidList.append(hidError)
    return hidList

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

def updateOutput(error, hiddenOutput):
    for i in error:
        bias2[error.index(i)] = bias2[error.index(i)]+ i*learningRate #New bias = old bias + the error of that node * learning rate
        for y in hiddenOutput:
            x=weights2[hiddenOutput.index(y)][error.index(i)]
            weights2[hiddenOutput.index(y)][error.index(i)] = x + (learningRate*i*y) #New weight = old weight + the error of that node * learning rate*the input from previous node

def updateHidden(error, row):
    input=[]
    for i in range(3):
        input.append(correct[row][i])
    for x in error:
        bias1[error.index(x)] = bias1[error.index(x)]+ x*learningRate
        for y in input:
           z = weights1[input.index(y)][error.index(x)]
           weights1[input.index(y)][error.index(x)] = z + (learningRate*x*y)


def main():
    #for n in range(1000): #Option for multiple epochs
        SSE =0
        for i in range(314):#iterates over every set
            layer1, layer2= forward(i)
            answer = get_correct(i)
            errorO, SSE = calculate_outputError(answer, layer2, SSE)
            errorH = calculate_hiddenError(errorO, layer1)
            updateOutput(errorO, layer1)
            updateHidden(errorH, i)
            print(errorO)
            print(errorH)
       # n = n +1
        SSE = 0 #calculate sse only after weights have been updated
        for i in range(314):
            layer1, layer2= forward(i)
            answer = get_correct(i)
            errorO, SSE = calculate_outputError(answer, layer2, SSE)
            errorH = calculate_hiddenError(errorO, layer1)
        print(weights1)
        print(weights2)
        print(SSE)
main()