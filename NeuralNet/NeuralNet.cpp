//
//  NeuralNet.cpp
//  NeuralNet
//
//  Created by Blake Jones on 12/30/17.
//  Copyright © 2017 Blake Jones. All rights reserved.
//

#include "NeuralNet.hpp"

NeuralNet::NeuralNet(vector<int> layerSize) {
    this->layerSizes = layerSize;
    numberHiddenLayers = layerSize.size() - 2;
    numberLayers = layerSize.size() + 1;
    
    for (int i = 0; i < numberHiddenLayers; i++) {
        for (int j = 0; j < layerSizes[i + 1]; j++) {
            vector<int> empty;
            Perceptron perc = Perceptron(layerSizes[i], empty);
            hiddenLayers[i].push_back(perc);
        }
    }
    
    for (int i = 0; i < layerSize.back(); i++) {
        vector<int> empty;
        Perceptron perc = Perceptron(layerSize[layerSize.size() - 2], empty);
        outputLayer.push_back(perc);
    }
    
    for (int i = 0; i < numberHiddenLayers; i++) {
        layers.push_back(hiddenLayers[i]);
    }
    layers.push_back(outputLayer);
}

NeuralNet::~NeuralNet() {
    
}
