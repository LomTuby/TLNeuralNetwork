//
//  TLNeuralNetwork.swift
//  Toms Neural Net
//
//  Created by Thomas Luby on 2/1/16.
//  Copyright © 2016 Thomas Luby. All rights reserved.
//

import Foundation

class TLNeuralNetwork: NSObject {
    
    var input = [Double]()
    var weight = [Double]()
    var bias = [Double]()
    var targetOutput = [Double]()
    var learningRate = 0.5
    
    // Neurons
    private var neuronHidden = [Neuron]()
    private var neuronOutput = [Neuron]()
    
    // Configuration
    private var numInputs = 0
    private var numHidden = 0
    private var numOutputs = 0
    
    init(numInputs: Int, numHidden: Int, numOutputs: Int) {
        // Number of input neurons depend on the number of inputs
        self.numInputs = numInputs
        self.numHidden = numHidden
        self.numOutputs = numOutputs
        
        let neuronHiddenArray = (0..<numHidden).map { _ in Neuron(numInputs: numInputs)}
        let neuronOutputArray = (0..<numOutputs).map { _ in Neuron(numInputs: numHidden)}
        
        self.neuronHidden = neuronHiddenArray
        self.neuronOutput = neuronOutputArray
    }
    
    // Forward pass
    var forwardPass: [Double] {
        
        get {
            
            var output = [Double]()
            var hidden = [Double]()
            
            // Loop through each hidden neuron
            for ptrNeuronHidden in 0..<self.numHidden {
                
                // All inputs go to all hidden neurons
                let inputStartIndex = 0
                let inputEndIndex = self.numInputs - 1
                
                // Map the correct weights to the hidden neurons
                let weightStartIndex = ptrNeuronHidden * self.numInputs
                let weightEndIndex = weightStartIndex + self.numInputs - 1
                
                neuronHidden[ptrNeuronHidden].input = Array(self.input[inputStartIndex...inputEndIndex])
                neuronHidden[ptrNeuronHidden].weight = Array(self.weight[weightStartIndex...weightEndIndex])
                
                // bias[0] for the hidden neurons, bias[1] for the output neurons
                neuronHidden[ptrNeuronHidden].bias = self.bias[0]
                
                hidden.append(neuronHidden[ptrNeuronHidden].forwardPass)
            }
            
        
            // Loop through each output neuron
            for ptrNeuronOutput in 0..<self.numOutputs {
                
                // Map the correct weights to the hidden neurons
                let weightStartIndex = (self.numHidden * self.numInputs) + ptrNeuronOutput * self.numHidden
                let weightEndIndex = weightStartIndex + self.numHidden - 1
                
                neuronOutput[ptrNeuronOutput].input = hidden
                neuronOutput[ptrNeuronOutput].weight = Array(self.weight[weightStartIndex...weightEndIndex])
                
                // bias[1] for the hidden neurons, bias[1] for the output neurons
                neuronOutput[ptrNeuronOutput].bias = self.bias[1]
                
                output.append(neuronOutput[ptrNeuronOutput].forwardPass)
                
            }

            
           return output
        }
        
    }
    
    // Update the weights
    func updateWeights() -> [Double] {
        
        var backwardPass = self.backwardPass
        
        for (index,weight) in self.weight.enumerate() {
            
            self.weight[index] = weight - self.learningRate * backwardPass[index]
        }
        
        return self.weight
    }
    
    // Backward pass
    var backwardPass: [Double] {
        
        get {
            let forwardPass = self.forwardPass
            var output = [Double]()
            
            var dEn_dOn = [Double]()
            var dOn = [[Double]]()
            var dHn = [[Double]]()
            
            /* Output weights
            
            eg for a 2 input 2 hidden network:
            
            let dE1_dW5 = dE1_dO1 * dO1_dW5
            let dE1_dW6 = dE1_dO1 * dO1_dW6
            let dE2_dW7 = dE2_dO2 * dO2_dW7
            let dE2_dW8 = dE2_dO2 * dO2_dW8
            */
            for ptrOutput in 0..<forwardPass.count {
                
                // Sensitivity of error to output
                dEn_dOn.append(-(targetOutput[ptrOutput] - forwardPass[ptrOutput]))
                
                // Sensitivity of output to inputs and weights
                dOn.append(self.neuronOutput[ptrOutput].backwardPass)
                
                for ptrHidden in 0..<self.numHidden {
                    // dEn_dWm where dWm are hidden weights = dEn_dOn * dOn_dwm
                    output.append(dEn_dOn[ptrOutput] * dOn[ptrOutput][self.numInputs+ptrHidden])
                }
            }
            

            
            /* Input weights
            
            eg for a 2 input 2 hidden network:

            let dE1_dW1 = dE1_dO1 * dO1_dH1 * dH1_dW1
            let dE2_dW1 = dE2_dO2 * dO2_dH1 * dH1_dW1
            let dE1_dW2 = dE1_dO1 * dO1_dH1 * dH1_dW2
            let dE2_dW2 = dE2_dO2 * dO2_dH1 * dH1_dW2
            let dE1_dW3 = dE1_dO1 * dO1_dH2 * dH2_dW3
            let dE2_dW3 = dE2_dO2 * dO2_dH2 * dH2_dW3
            let dE1_dW4 = dE1_dO1 * dO1_dH2 * dH2_dW4
            let dE2_dW4 = dE2_dO2 * dO2_dH2 * dH2_dW4

            */
            
            
            for ptrHidden in 0..<self.numHidden  {
              
                dHn.append(self.neuronHidden[ptrHidden].backwardPass)
                
                for ptrInputWeight in 0..<(self.numInputs) {
                    
                    var eTotal = 0.0
                    for ptrOutput in 0..<forwardPass.count {
                        
                        eTotal += dEn_dOn[ptrOutput] * dOn[ptrOutput][ptrHidden] * dHn[ptrHidden][self.numHidden+ptrInputWeight]
                    }
                    
                    output.append(eTotal)
                }
            }
            
            // Flip output array so sensitivities to weights are inputs first then outputs
            let outputFlipped = Array(output[(self.weight.count)/2...self.weight.count-1]) + Array(output[0...(self.weight.count)/2-1])
            return outputFlipped
        }
        
    }
    
    // Calculate mean squared error for each output and return the sum
    var msError: [Double] {
        
        get {
            let forwardPass = self.forwardPass
            var errorTotal = [Double]()
            
            for ptrOutput in 0..<forwardPass.count {
                errorTotal.append(0.5 * pow((self.targetOutput[ptrOutput]-forwardPass[ptrOutput]),2))
                
            }
            
            return errorTotal
        }
    }
    
    
}