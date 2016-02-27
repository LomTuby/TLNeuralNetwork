# TLNeuralNetwork

This is a pure Swift implementation of a two layer, multiple input / hidden / output node neural network. Currently it supports the sigmoid activation function

Example code to get started:

let tlnn = TLNeuralNetwork(numInputs: 2, numHidden: 2, numOutputs: 2)
        
        tlnn.input = [0.05, 0.1]
        tlnn.weight = [0.15,0.2,0.25,0.3,0.4,0.45,0.5,0.55]
        tlnn.bias = [0.35,0.6]
        tlnn.targetOutput = [0.01,0.99]
        tlnn.learningRate = 0.5
        
        print("Forward pass: O1,O2",tlnn.forwardPass)
        
        // 10 epochs
        for (var i=0; i<10; i++) {
            tlnn.updateWeights()
        }
        
        print("After 10 epochs: O1,O2",tlnn.forwardPass)
        
  The input array needs to have numInputs of values
  
  The weight array is defined as follows: The first (numInputs x numHidden) values are the weights for the hidden nodes while
  the last (numHidden x numOutputs) values are the weights for the output nodes. Weights are ordered as 
  
  [input 1 weight for hidden node 1, input 2 weight for hidden node 1, input 1 weight for hidden node 2 etc]
  
  Test cases included for 2x2x2 and 3x3x3 networks
  
