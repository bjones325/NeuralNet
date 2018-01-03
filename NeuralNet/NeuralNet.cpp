//
//  NeuralNet.cpp
//  NeuralNet
//
//  Created by Blake Jones on 12/30/17.
//  Copyright © 2017 Blake Jones. All rights reserved.
//

#include "NeuralNet.hpp"

NeuralNet::NeuralNet(vector<int> &layerSize) {
    this->layerSizes = layerSize;
    numberHiddenLayers = layerSize.size() - 2;
    numberLayers = layerSize.size() + 1;
    
    for (int i = 0; i < numberHiddenLayers; i++) {
        for (int j = 0; j < layerSizes[i + 1]; j++) {
            vector<float> empty;
            Perceptron perc = Perceptron(layerSizes[i], empty);
            hiddenLayers[i].push_back(perc);
        }
    }
    
    for (int i = 0; i < layerSize.back(); i++) {
        vector<float> empty;
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

vector<vector<float>> NeuralNet::feedForward(vector<float> &inActuals) {
    vector<vector<float>> inputList;
    inputList.push_back(inActuals);
    for (int i = 0; i < numberLayers; i++) {
        vector<Perceptron> percep = layers[i];
        vector<float> currentInput;
        for (int j = 0; j < percep.size(); j++) {
            Perceptron currentPerc = percep[j];
            vector<float> input = inputList[i];
            currentInput.push_back(currentPerc.sigmoidActivation(input));
        }
        inputList.push_back(currentInput);
    }
    return inputList;
}

tuple<float> NeuralNet::backPropLearning(vector<Example> examples, int alpha) {
    
    float averageError = 0;
    float averageWeightChange = 0;
    float numWeights = 0;
    
    for (int i = 0; i < examples.size(); i++) {
        vector<float> deltas;
    }
}
