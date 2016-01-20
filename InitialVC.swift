//
//  InitialVC.swift
//  Toms Neural Net
//
//  Created by Thomas Luby on 29/12/15.
//  Copyright © 2015 Thomas Luby. All rights reserved.
//

import UIKit

class InitialVC: UIViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
    
        let tlnn = TLNeuralNetwork(numInputs: 2, numHidden: 2, numOutputs: 2)
        
        tlnn.input = [0.05, 0.1]
        tlnn.weight = [0.15,0.2,0.25,0.3,0.4,0.45,0.5,0.55]
        tlnn.bias = [0.35,0.6]
        tlnn.targetOutput = [0.01,0.99]
        tlnn.bias = [0.35,0.6]
        tlnn.learningRate = 0.5
        
        for (var i = 0; i<10000; i++) {
            tlnn.updateWeights()
        }
        
        print("O1,O2",tlnn.forwardPass)
        
    }




}

