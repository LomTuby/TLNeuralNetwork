//
//  TLNeuralNetworkTests.swift
//  TLNeuralNetworkTests
//
//  Created by Thomas Luby on 20/1/16.
//  Copyright © 2016 Thomas Luby. All rights reserved.
//

import XCTest
@testable import TLNeuralNetwork

extension Double {
    /// Rounds the double to decimal places value
    func roundToPlaces(places:Int) -> Double {
        let divisor = pow(10.0, Double(places))
        return round(self * divisor) / divisor
    }
}

class TLNeuralNetworkTests: XCTestCase {
    
    var tlnn : TLNeuralNetwork!
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
        
        self.tlnn = TLNeuralNetwork(numInputs: 2, numHidden: 2, numOutputs: 2)
        self.tlnn.input = [0.05, 0.1]
        self.tlnn.weight = [0.15,0.2,0.25,0.3,0.4,0.45,0.5,0.55]
        self.tlnn.bias = [0.35,0.6]
        self.tlnn.targetOutput = [0.01,0.99]
        self.tlnn.bias = [0.35,0.6]
        self.tlnn.learningRate = 0.5
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        self.tlnn = nil
        super.tearDown()
    }
    
    func testTwoInputTwoHiddenNN() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        
        var fwdPass = tlnn.forwardPass

        XCTAssertEqual(fwdPass[0].roundToPlaces(4), (0.751365069552316).roundToPlaces(4), "O1 does not equal 0.751365069552316")
        XCTAssertEqual(fwdPass[1].roundToPlaces(4), (0.772928465321463).roundToPlaces(4), "O2 does not equal 0.772928465321463")
        
        var newWeights = self.tlnn.updateWeights()
        
        XCTAssertEqual(newWeights[0].roundToPlaces(4), (0.149780716132763).roundToPlaces(4), "Error with w1")
        XCTAssertEqual(newWeights[1].roundToPlaces(4), (0.199561432265526).roundToPlaces(4), "Error with w2")
        XCTAssertEqual(newWeights[2].roundToPlaces(4), (0.24975114363237).roundToPlaces(4), "Error with w3")
        XCTAssertEqual(newWeights[3].roundToPlaces(4), (0.299502287264739).roundToPlaces(4), "Error with w4")
        XCTAssertEqual(newWeights[4].roundToPlaces(4), (0.358916479717885).roundToPlaces(4), "Error with w5")
        XCTAssertEqual(newWeights[5].roundToPlaces(4), (0.408666186076233).roundToPlaces(4), "Error with w6")
        XCTAssertEqual(newWeights[6].roundToPlaces(4), (0.511301270238738).roundToPlaces(4), "Error with w7")
        XCTAssertEqual(newWeights[7].roundToPlaces(4), (0.561370121107989).roundToPlaces(4), "Error with w8")
        
        for (var i = 0; i<10000; i++) {
            tlnn.updateWeights()
        }
        
        fwdPass = tlnn.forwardPass
        
        XCTAssertEqual(fwdPass[0].roundToPlaces(4), (0.0159130559446298).roundToPlaces(4), "O1 does not equal 0.0159130559446298")
        XCTAssertEqual(fwdPass[1].roundToPlaces(4), (0.984064834634694).roundToPlaces(4), "O2 does not equal 0.984064834634694")
        
    }
    
    
    func testPerformance() {
        // This is an example of a performance test case.
        self.measureBlock {
            // Put the code you want to measure the time of here.
        }
    }
    

    
}
