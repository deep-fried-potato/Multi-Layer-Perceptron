import math
import random
import datetime
class DataPoint:
    def __init__(self,attributes):
        self.attributes = [1] + attributes[:-1]
        self.label = attributes[-1]
    def __str__(self):
        return str(self.attributes) + " : " +str(self.label)

def log_output(string):
    f = open("logfile_SSD.txt","a")
    f.write("\n"+str(datetime.datetime.now()) +" : "+ string+"\n")

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def differentiated_sigmoid(x):
  return sigmoid(x)*(1-sigmoid(x))

def addlists(list1,list2):
    return [x1+x2 for (x1,x2) in zip(list1,list2)]

def scalarprod(scalar,vector):
    return [scalar*element for element in vector]

def dotprod(K, L):
    if len(K) != len(L):
        return 0
    return sum(i[0] * i[1] for i in zip(K, L))

def activation_sigmoid(weights,attributes):
    return sigmoid(dotprod(weights,attributes))

def discriminant(classlabel,DataPoint,input_weights,output_weights):
    hidden_layer_output = [activation_sigmoid(input_weights[neuron],DataPoint.attributes) for neuron in range(N_neurons)]
    return activation_sigmoid(output_weights[classlabel],hidden_layer_output)

def del_k(classlabel,DataPoint,input_weights,output_weights,pointcount,delta_k_global):
    hidden_layer_output = [activation_sigmoid(input_weights[neuron],DataPoint.attributes) for neuron in range(N_neurons)]
    netk = dotprod(output_weights[classlabel],hidden_layer_output)
    del_k = (int((DataPoint.label-1)==classlabel)/discriminant(classlabel,DataPoint,input_weights,output_weights))*differentiated_sigmoid(netk)
    delta_k_global[pointcount][classlabel]=del_k
    return del_k

def change_output_weight(classlabel,TrainSet,input_weights,output_weights,delta_k_global):
    output_weight_result = [0]*N_neurons
    pointcount=0
    for DataPoint in TrainSet:
        delta_k = del_k(classlabel,DataPoint,input_weights,output_weights,pointcount,delta_k_global)
        hidden_layer_output = [activation_sigmoid(input_weights[neuron],DataPoint.attributes) for neuron in range(N_neurons)]
        output_weight_result = addlists(output_weight_result,scalarprod(delta_k,hidden_layer_output))
        pointcount+=1
    return output_weight_result

def del_j(neuron,DataPoint,input_weights,output_weights,delta_k_global,pointcount):
    sum = 0
    for classlabel in range(N_classes):
        netj = dotprod(DataPoint.attributes,input_weights[neuron])
        sum += delta_k_global[pointcount][classlabel]*output_weights[classlabel][neuron]*differentiated_sigmoid(netj)
    return sum
def change_input_weight(neuron,TrainSet,input_weights,output_weights,delta_k_global):
    input_weight_result = [0]*len(TrainSet[0].attributes)
    pointcount=0
    for DataPoint in TrainSet:
        delta_j = del_j(neuron,DataPoint,input_weights,output_weights,delta_k_global,pointcount)
        input_weight_result += scalarprod(delta_j,DataPoint.attributes)
        pointcount+=1
    return input_weight_result

def train(TrainSet,input_weights,output_weights):
    eta = 0.001
    epoch = 0
    delta_k_global = [[0]*N_classes]*len(TrainSet)
    while(epoch < 10):
        prev_out_weights = output_weights
        output_weights = [ addlists(output_weights[classlabel],scalarprod(eta,change_output_weight(classlabel,TrainSet,input_weights,output_weights,delta_k_global))) for classlabel in range(N_classes)]
        input_weights = [addlists(input_weights[neuron],scalarprod(eta,change_input_weight(neuron,TrainSet,input_weights,prev_out_weights,delta_k_global))) for neuron in range(N_neurons)]
        string = "Iteration:" + str(epoch) + " Error:"+ str(test(TrainSet,input_weights,output_weights))
        log_output(string)
        print(string)
        epoch+=1
    return [input_weights,output_weights]
def test(TestSet,input_weights,output_weights):
    sum_error = 0
    for DataPoint in TestSet:
        for classlabel in range(N_classes):
            sum_error += (int((DataPoint.label-1)==classlabel)-discriminant(classlabel,DataPoint,input_weights,output_weights))**2
    return sum_error/(2*len(TestSet)*N_classes)
#-----------------------------------------------

fileobj = open("Colon_Cancer_CNN_Features.csv","r")
filedata = fileobj.readlines()
DataSet = [DataPoint([float(x) for x in line.split(",")]) for line in filedata if line.split()]
random.shuffle(DataSet)
TrainSet = DataSet[:5000]
TestSet = DataSet[5000:]

for i in range(5,15):
    string = str(i) + " Neurons"
    log_output(string)
    print(string)
    N_neurons = i
    N_classes = 4
    input_weights = [[1]*len(DataSet[0].attributes)]*N_neurons
    output_weights = [[1]*N_neurons]*N_classes
    [trained_input_weights,trained_output_weights]=train(DataSet,input_weights,output_weights)
    test_error = test(TestSet,trained_input_weights,trained_output_weights)
    string = "Test error: " + str(test_error)
    log_output(string)
    print(string)
