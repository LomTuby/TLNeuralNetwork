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
 
        print("O1,O2",tlnn.forwardPass)
        print("New weights", tlnn.updateWeights())

   
        
        print ("Control: [0.149780716132763, 0.199561432265526, 0.24975114363237, 0.299502287264739, 0.358916479717885, 0.408666186076233, 0.511301270238738, 0.561370121107989]")
        
        print("O1,O2",tlnn.forwardPass)
        print("New weights", tlnn.updateWeights())
    }




}

