//
//  InitialVC.swift
//  Toms Neural Net
//
//  Created by Thomas Luby on 29/12/15.
//  Copyright © 2015 Thomas Luby. All rights reserved.
//

// Input array of size numInputs contains the values for each input node
// Weight array of size (numInputs x numHidden) contains the weights for each input per hidden node ie for a 2 input 2 hidden node array, the first two weights are related to input 1 and 2 for hidden node 1
// Bias array contains optional bias values for the hidden nodes and the output nodes. To ignore bias set this to [1.0,1.0]

import UIKit

class InitialVC: UIViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
    
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
        
        let tlnn2 = TLNeuralNetwork(numInputs: 3, numHidden: 3, numOutputs: 3)
        
        tlnn2.input = [0.05, 0.1, 0.1]
        tlnn2.weight = [0.15, 0.2, 0.25, 0.3, 0.35, 0.3, 0.25, 0.2, 0.15, 0.15, 0.2, 0.25, 0.3, 0.35, 0.3, 0.25, 0.2, 0.15]
        tlnn2.bias = [0.35,0.6]
        tlnn2.targetOutput = [0.01,0.99, 0.99]
        tlnn2.learningRate = 0.5
        
        print("Forward pass:O1,O2",tlnn2.forwardPass)
        
        // 10 epochs
        for (var i=0; i<10; i++) {
            tlnn2.updateWeights()
        }
        
        print("After 10 epochs:O1,O2",tlnn2.forwardPass)
        
    }




}

