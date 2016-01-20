//
//  Neuron.swift
//  Toms Neural Net
//
//  Created by Thomas Luby on 29/12/15.
//  Copyright © 2015 Thomas Luby. All rights reserved.
//

import Foundation

class Neuron: NSObject {
    
    // Inputs
    var input = [Double]()
    var weight = [Double]()
    var bias: Double = 0.0
    
    // Nodes
    private var mult = [Mult]()
    private var add = Add()
    private var activation = Sigmoid()

    // Initialiser
    init(numInputs: Int) {
        // Number of multiply blocks depends on the number of inputs
        let multArray = (0..<numInputs).map { _ in Mult()}
        self.mult = multArray
    }
    
    // Forward pass
    var forwardPass: Double {
        
        get {
            
            for (index, ptrInput) in self.input.enumerate() {
                self.mult[index].input = Array([ptrInput, self.weight[index]])
            }
            
            self.add.input = [Double]()
            
            for ptrMult in self.mult {
                self.add.input.append(ptrMult.forwardPass)
            }
            
            self.add.input.append(self.bias)
            
            self.activation.input = add.forwardPass
            return activation.forwardPass
        }
    }
    
    // Backward pass, derivative wrt weight
    var backwardPass: [Double] {
        
        get {
            
            let actAddDeriv = activation.backwardPass * add.backwardPass
            var bPass = [Double]()
            
            for ptrbPass in 0...1 {
                for value in self.mult {
                    bPass.append(actAddDeriv * value.backwardPass[ptrbPass])
                }
            }
            
            return bPass
        }
    }
    
    
    class Mult: NSObject {
        
        // Inputs
        var input = [Double]()
        
        // Forward pass
        var forwardPass: Double {
            
            get {
                return self.input.reduce(1, combine: *)
            }
        }
        
        // Backward pass
        var backwardPass: [Double] {
            
            get {
                
                let product = self.input.reduce(1, combine: *)
                return self.input.map{product / $0}
            }
        }
        
    }
    
    class Add: NSObject {
        
        // Inputs
        var input = [Double]()
        var bias: Double = 0.0
        
        // Forward pass
        var forwardPass: Double {
            
            get {
                return self.input.reduce(0, combine: +) + self.bias
            }
        }
        
        // Backward pass
        var backwardPass: Double {
            
            get {
                return 1
            }
        }
        
    }
    
    class Sigmoid: NSObject {
        
        // Input
        var input: Double = 0.0
        
        // Forward pass
        var forwardPass: Double {
            
            get {
                return sigmoid(self.input)
            }
        }
        
        // Backward pass
        var backwardPass: Double {
            
            get {
                return sigmoid(self.input) * (1-sigmoid(self.input))
            }
        }
        
        private func sigmoid(x: Double) -> Double {
            return 1 / (1 + exp(-x))
        }
    }
    
    
}


