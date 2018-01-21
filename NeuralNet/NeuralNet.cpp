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

tuple<float, float> NeuralNet::backPropLearning(vector<Example> examples, float alpha) {
    
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
        
        for (int i = (int) numberHiddenLayers - 1; i > -1; i--) {
            vector<Perceptron> layer = layers[i];
            vector<Perceptron> nextLayer = layers[i+1];
            vector<float> hiddenDelta;
            
            for (int j = 0; j < layer.size(); j++) {
                float gPrime = layer[j].sigmoidActivationDeriv(allLayerOutput[i]);
                float delta = 0.0;
                
                for (int nextLayInd = 0; i < nextLayer.size(); nextLayInd++) {
                    vector<float> percepWeight = nextLayer[nextLayInd].getWeights();
                    percepWeight.erase(percepWeight.begin() - 1);
                    delta += (percepWeight[i] * deltas[0][j]);
                }
                delta = gPrime * delta;
                hiddenDelta.push_back(delta);
            }
            deltas.insert(deltas.begin(), hiddenDelta);
        }
        
        for (int i = 0; i < numberLayers; i++) {
            vector<Perceptron> layer = layers[i];
            for (int j = 0; i < layer.size(); j++) {
                float weightMod = layer[j].updateWeights(allLayerOutput[i], alpha, deltas[i][j]);
                averageWeightChange += weightMod;
                numWeights += layer[j].getInputSize();
            }
        }
    }
    averageError /= (examples.size() * examples[0].getInputList().size());
    averageWeightChange /= numWeights;
    tuple<float, float> results (averageError, averageWeightChange);
    return results;
}

void trainNeuralNet(vector<Example> examples, vector<Example> test, float alpha, float weightChangeThreshold, NeuralNet net) {
    int numberIn = examples[0].getInputList().size();
    int numberOut = examples[0].getOutputList().size();
    
    int Time = 5;
    vector<int> hiddenLayerList;
    for (vector<Perceptron> layer : net.hiddenLayers) {
        hiddenLayerList.push_back((int) layer.size());
    }
    cout << "Beginning training of NeuralNet";
    
    /*vector<int> layerList = hiddenLayerList;
    layerList.insert(layerList.begin(), numberIn);
    layerList.insert(layerList.end(), numberOut);*/
    
    int iteration = 0.0;
    float trainError = 0.0;
    float weightMod = 0.0;
    
    tuple<float, float> results = net.backPropLearning(examples, alpha);
    trainError = get<0>(results);
    weightMod = get<1>(results);
    iteration++;
    while (weightMod >= weightChangeThreshold and iteration <= INT_MAX) {
        tuple<float, float> nextResults = net.backPropLearning(examples, alpha);
        float trainError = get<0>(nextResults);
        float weightMod = get<1>(nextResults);
        iteration++;
    }
    
    cout << "Finished after " << iteration << " iterations. Good job.";
    cout << "Train Error: " << trainError << " Weight Mod: " << weightMod;
    cout << endl << "Examining accuracy of neural network.";
    
    float testError = 0.0;
    float testCorrect = 0.0;
    float testAccuracy = 0.0;
    
    for (Example exam : test) {
        vector<float> inputList = exam.getInputList();
        vector<float> outputList = exam.getOutputList();
        
        vector<vector<float>> results = net.feedForward(inputList);
        vector<float> roundedResults;
        
        for (int i = 0; i < results[results.size() - 1].size(); i++) {
            float item = results[results.size() - 1][i];
            roundedResults.push_back(round(item));
        }
        
        if (outputList == roundedResults) {
            testCorrect++;
        } else {
            testError++;
        }
    }
    
    float testTotal = testCorrect + testError;
    testAccuracy = testCorrect / testTotal;
    
    cout << endl << "Feed Forward Test Correctly Classified " << testCorrect << ", incorrectly classified " << testError << " for an accuracy of " << testAccuracy;
}
