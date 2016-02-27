//
//  TLNeuralNetwork.swift
//  Toms Neural Net
//
//  Created by Thomas Luby on 2/1/16.
//  Copyright © 2016 Thomas Luby. All rights reserved.
//

import Foundation

public class TLNeuralNetwork: NSObject {
    
    private let numLayers = 2
    
    var input: [Double] {
        set {
            if self.numInputs > newValue.count {
                print("TLNeuralNetwork: Warning input has " + String(self.numInputs) + " inputs specified but only " + String(newValue.count) + " set. Setting remaining to zero")
                
                var userRequestedInput = newValue
                // Pad with zeros
                userRequestedInput += [Double](count: (self.numInputs - newValue.count), repeatedValue: 0.0)
                self._input = userRequestedInput
                
            } else if self.numInputs < newValue.count {
                print("TLNeuralNetwork: Warning input has " + String(self.numInputs) + " inputs specified but " + String(newValue.count) + " set. Will ignore the last " + String(newValue.count - self.numInputs))
          
                var userRequestedInput = newValue
                self._input = Array(userRequestedInput[0..<self.numInputs])
            } else {
                self._input = newValue
            }
        }
        get {
            return _input
        }
    }
    
    var weight: [Double] {
        set {
            let numWeights = (self.numLayers * self.numHidden * self.numOutputs)
            
            if numWeights > newValue.count {
                print("TLNeuralNetwork: Warning weight has " + String(numWeights) + " weights specified but only " + String(newValue.count) + " set. Setting remaining to zero")
                
                var userRequestedInput = newValue
                // Pad with zeros
                userRequestedInput += [Double](count: (numWeights - newValue.count), repeatedValue: 0.0)
                self._weight = userRequestedInput
                
            } else if numWeights < newValue.count {
                print("TLNeuralNetwork: Warning weight has " + String(numWeights) + " weights specified but " + String(newValue.count) + " set. Will ignore the last " + String(newValue.count - numWeights))
                
                var userRequestedInput = newValue
                self._weight = Array(userRequestedInput[0..<numWeights])
            } else {
                self._weight = newValue
            }
        }
        get {
            return _weight
        }
    }
    
    var bias: [Double] {
        set {
            if self.numLayers > newValue.count {
                print("TLNeuralNetwork: Warning bias has " + String(self.numLayers) + " biases expected but only " + String(newValue.count) + " set. Setting remaining to 1.0")
                
                var userRequestedInput = newValue
                // Pad with 1.0
                userRequestedInput += [Double](count: (self.numLayers - newValue.count), repeatedValue: 1.0)
                self._bias = userRequestedInput
                
            } else if self.numLayers < newValue.count {
                print("TLNeuralNetwork: Warning bias has " + String(self.numLayers) + " biases expected but " + String(newValue.count) + " set. Will ignore the last " + String(newValue.count - self.numLayers))
                
                var userRequestedInput = newValue
                self._bias = Array(userRequestedInput[0..<self.numLayers])
            } else {
                self._bias = newValue
            }
        }
        get {
            return _bias
        }
    }
    
    var targetOutput: [Double] {
        set {
            if self.numOutputs > newValue.count {
                print("TLNeuralNetwork: Warning targetOutput has " + String(self.numOutputs) + " inputs specified but only " + String(newValue.count) + " set. Setting remaining to zero")
                
                var userRequestedInput = newValue
                // Pad with zeros
                userRequestedInput += [Double](count: (self.numOutputs - newValue.count), repeatedValue: 0.0)
                self._targetOutput = userRequestedInput
                
            } else if self.numOutputs < newValue.count {
                print("TLNeuralNetwork: Warning targetOutput has " + String(self.numOutputs) + " inputs specified but " + String(newValue.count) + " set. Will ignore the last " + String(newValue.count - self.numOutputs))
                
                var userRequestedInput = newValue
                self._targetOutput = Array(userRequestedInput[0..<self.numOutputs])
            } else {
                self._targetOutput = newValue
            }
        }
        get {
            return _targetOutput
        }
    }
    
    private var _input = [Double]()
    private var _weight = [Double]()
    private var _bias = [Double]()
    private var _targetOutput = [Double]()
    var learningRate = 0.5
    
    // Neurons
    private var neuronHidden = [Neuron]()
    private var neuronOutput = [Neuron]()
    
    // Configuration
    private var numInputs = 0
    private var numHidden = 0
    private var numOutputs = 0
    
    init(numInputs: Int, numHidden: Int, numOutputs: Int) {
        super.init()
        
        // Number of input neurons depend on the number of inputs
        self.numInputs = numInputs
        self.numHidden = numHidden
        self.numOutputs = numOutputs
        
        let neuronHiddenArray = (0..<numHidden).map { _ in Neuron(numInputs: numInputs)}
        let neuronOutputArray = (0..<numOutputs).map { _ in Neuron(numInputs: numHidden)}
        
        self.neuronHidden = neuronHiddenArray
        self.neuronOutput = neuronOutputArray
        
        // Set defaults
        self.input = [Double](count: numInputs, repeatedValue: 0.0)
        self.weight = [Double](count: (self.numLayers * self.numHidden * self.numOutputs), repeatedValue: 0.0)
        self.bias = [1.0,1.0]
        
    }
    
    // Forward pass
    var forwardPass: [Double] {
        
        get {
            
            var output = [Double]()
            var hidden = [Double]()
            
            // All inputs go to all hidden neurons
            let inputStartIndex = 0
            let inputEndIndex = self.numInputs - 1
            
            // Loop through each hidden neuron
            for ptrNeuronHidden in 0..<self.numHidden {
                
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
        
        for index in 0..<self.weight.count {
            
            self.weight[index] -= self.learningRate * backwardPass[index]
        }
        
        return self.weight
    }
    
    // Backward pass for the error function
    private func mseBackwardPass(ptrOutput: Int) -> Double {
        return -(targetOutput[ptrOutput] - forwardPass[ptrOutput])
    }
    
    // Backward pass
    var backwardPass: [Double] {
        
        get {
            let forwardPass = self.forwardPass
            var output = [Double]()
            
            var dEn_dOn = [Double]()
            var dOn = [[Double]]()
            
            /* Output weights
            
            eg for a 2 input 2 hidden network:
            
            let dE1_dW5 = dE1_dO1 * dO1_dW5
            let dE1_dW6 = dE1_dO1 * dO1_dW6
            let dE2_dW7 = dE2_dO2 * dO2_dW7
            let dE2_dW8 = dE2_dO2 * dO2_dW8
            */
            
            for ptrOutput in 0..<forwardPass.count {
                
                // Sensitivity of error to output
                dEn_dOn.append(mseBackwardPass(ptrOutput))
                
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
              
                var dHn = self.neuronHidden[ptrHidden].backwardPass
                
                // self.numInouts number of weights for each hidden neuron
                for ptrInputWeight in 0..<(self.numInputs) {
                    
                    var eTotal = 0.0
                    for ptrOutput in 0..<forwardPass.count {
                        
                        eTotal += dEn_dOn[ptrOutput] * dOn[ptrOutput][ptrHidden] * dHn[self.numHidden+ptrInputWeight]
                    }
                    
                    output.append(eTotal)
                }
            }
            
            // Flip output array so sensitivities to weights are inputs first then outputs
            let outputFlipped = Array(output[(self.weight.count)/2...self.weight.count-1]) + Array(output[0...(self.weight.count)/2-1])
            return outputFlipped
        }
        
    }
    
    
    
}