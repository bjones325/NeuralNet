//
//  NeuralNet.cpp
//  NeuralNet
//
//  Created by Blake Jones on 12/30/17.
//  Copyright © 2017 Blake Jones. All rights reserved.
//

#include "NeuralNet.hpp"

NeuralNet::NeuralNet(int layerSize) {
    this->layerSize = layerSize;
    numberHiddenLayers = layerSize - 2;
    numberLayers = numberHiddenLayers + 1
<<<<<<< HEAD
<<<<<<< HEAD
    outputLayer = {};
=======
    outputLayer = {}
>>>>>>> d4bb09e... Initial Commit - Finished Perceptron, beginning NeuralNet
=======
    outputLayer = {}
>>>>>>> d4bb09e... Initial Commit - Finished Perceptron, beginning NeuralNet
}
