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

vector<vector<float>> NeuralNet::feedForward(const vector<float> &inActuals) {
    vector<vector<float>> inputList;
    inputList.push_back(inActuals);
    for (int i = 0; i < numberLayers; i++) {
        vector<Perceptron> percep = layers[i];
        vector<float> currentInput;
        for (Perceptron currentPerc : percep) {
            vector<float> input = inputList[i];
        currentInput.push_back(currentPerc.sigmoidActivation(input));
        }
        inputList.push_back(currentInput);
    }
    return inputList;
}

tuple<float> NeuralNet::backPropLearning(vector<Example> examples, float alpha) {
    
    float averageError = 0;
    float averageWeightChange = 0;
    float numWeights = 0;
    
    for (Example currExample : examples) {
        vector<vector<float>> deltas;
        vector<float> outDelta;
        vector<vector<float>> allLayerOutput = feedForward(currExample.getInputList());
        vector<float> lastLayerOutput = allLayerOutput.back();
        
        for (int i = 0; i < currExample.getOutputList().size(); i++) {
            float gPrime = outputLayer[i].sigmoidActivation(allLayerOutput[allLayerOutput.size() - 2]);
            float error = currExample.getOutput(i) - lastLayerOutput[i];
            float delta = gPrime * error;
            averageError += error * error / 2;
            outDelta.push_back(delta);
        }
        deltas.push_back(outDelta);
        
        //Backprop hiddens
    }
}
