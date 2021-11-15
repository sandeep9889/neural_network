from typing import final
import  numpy
from numpy.core.fromnumeric import ndim
from numpy.lib.function_base import select
import scipy.special
# neural network class inisilization

class neuralnetwork:
    
    def __init__(self, inputnode, hiddennode, outputnode, learningrate):
        self.inodes = inputnode
        self.hnodes = hiddennode
        self.onodes = outputnode
        
        # learning rate
        self.lr = learningrate

        #weight inside the array are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w2 etc
        self.wih =(numpy.random.rand(self.hnodes , self.inodes) - 0.5)
        self.who =(numpy.random.rand(self.onodes , self.hnodes) - 0.5)

        # activation function is sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)


        pass
    
    def train(self, input_list, target_list):
        #convert input list into 2d array
        inputs =numpy.array(input_list, ndim =2).T
        targets = numpy.array(target_list, ndim =2).T

        #calculate signals into hidden layer 
        hidden_inputs =numpy.dot(self.wih, inputs)

        # claculate signals emerging from 1st layer

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who,hidden_outputs)

        final_outputs  = self.activation_function(final_inputs)
        #output layer error is the (target-actual)
        output_error = targets- final_outputs
        #hidden layer error is the output_error, split
        hidden_error = numpy.dot(self.who.T,output_error)

        #update the weight that link with  hidden and output layer
        self.who +=self.lr + numpy.dot((output_error * final_outputs * (1.0  - final_outputs)), numpy.transpose(hidden_outputs))
         
        # update the weight of input and hidden layer
        # its a equation for update the weight
        self.wih += self.lr *numpy.dot((hidden_error * hidden_outputs *(1.0-hidden_outputs)), numpy.transpose(inputs))
        pass
    
    def query(self,inputs_list):
        #convert input list into 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs =numpy.dot(self.wih,inputs)
        # calculate signals emerging from hidden layers 
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals of final output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)
        # calculate signals emerging from final output layer
        final_output = self.activation_function(final_inputs)

        return final_output




input_node = 3
hidden_node = 3
output_node = 3
learning_rate =0.3
n = neuralnetwork(input_node,hidden_node,output_node,learning_rate)

learning_rate = 0.3
n.query([1.0,0.5,-1.5])
