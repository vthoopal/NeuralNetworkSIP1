import numpy as np
import pandas as pd
weight1file = pd.read_excel(r"C:\Users\vedth\Desktop\sip\NeuralNetwork1\sip1\w1 (3 inputs - 11 nodes).xlsx")
weights1 = np.transpose(weight1file.to_numpy())
correctfile = pd.read_excel(r"C:\Users\vedth\Desktop\sip\NeuralNetwork1\sip1\cross_data (3 inputs - 2 outputs).xlsx")
correct = correctfile.to_numpy()
weight2file = pd.read_excel(r"C:\Users\vedth\Desktop\sip\NeuralNetwork1\sip1\w2 (from 11 to 2).xlsx")
weights2 = np.transpose(weight2file.to_numpy())
learningRate = 0.7
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

def get_correct(row):
    list=[]
    C1 = correct[row][3]
    C2= correct[row][4]
    list=[C1,C2]
    return list

def get_inputs(row):
    x1=correct[row][0]
    x2=correct[row][1]
    x3=correct[row][2]
    input_matrix =[x1,x2,x3]
    return input_matrix

def calculate_outputError(true, product):
    errorlist =[]
    for i in true:
       e = ((i - product[true.index(i)]) * activation_derivative(product[true.index(i)]))
       e = (int(e*10000))/10000
       errorlist.append(e)
    return errorlist

def calculate_hiddenError(finError, outputs):
    hidList=[]
    for y in outputs:
        w=0
        for i in finError:
            w = weights2[outputs.index(y)][finError.index(i)] + w
        hidError = w*activation_derivative(y)
        hidError =(int(hidError*10000)/10000)
        hidList.append(hidError)
    return hidList

def create_output(matrix, bias):
    solve=[]
    matrix = np.add(matrix, bias)
    for x in matrix:
        solve.append((int(activation(x)*10000))/10000) 
    return solve

def forward(row):
    hidden_value = np.dot(get_inputs(row), weights1)
    hidden_output=(create_output(hidden_value, bias1))

    final_value = np.dot(hidden_output, weights2)
    final_output = create_output(final_value, bias2)
    return hidden_output, final_output

def updateOutput(error, hiddenOutput):
    for i in error:
        bias2[error.index(i)] = bias2[error.index(i)]+ i*learningRate
        for y in hiddenOutput:
            x=weights2[hiddenOutput.index(y)][error.index(i)]
            weights2[hiddenOutput.index(y)][error.index(i)] = x + (learningRate*i*y)

def updateHidden(error, row):
    input=[]
    for i in range(3):
        input.append(correct[row][i])
    for x in error:
        bias1[error.index(x)] = bias1[error.index(x)]+ x*learningRate
        for y in input:
           z = weights1[input.index(y)][error.index(x)]
           weights1[input.index(y)][error.index(x)] = z + learningRate*x*y

    


def iterate():
    for i in range(314):
        layer1, layer2= forward(i)
        answer = get_correct(i)
        errorO = calculate_outputError(answer, layer2)
        errorH = calculate_hiddenError(errorO, layer1)
        updateOutput(errorO, layer1)
        updateHidden(errorH, i)
        print(errorO)
        print(errorH)
iterate()