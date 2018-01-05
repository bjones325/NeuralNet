//
//  NeuralNet.hpp
//  NeuralNet
//
//  Created by Blake Jones on 12/30/17.
//  Copyright © 2017 Blake Jones. All rights reserved.
//

#ifndef NeuralNet_hpp
#define NeuralNet_hpp

#include <stdio.h>
#include <tuple>
#include <climits>
#include "Perceptron.hpp"
#include "Example.hpp"

class NeuralNet {
private:
    unsigned long numberHiddenLayers;
    unsigned long numberLayers;
    vector<int> layerSizes;
    vector<Perceptron> outputLayer;
    vector<vector<Perceptron>> hiddenLayers;
    vector<vector<Perceptron>> layers;
    
public:
    NeuralNet(vector<int> &layerSize);
    ~NeuralNet();
    vector<vector<float>> feedForward(const vector<float> &inActuals);
    tuple<float, float> backPropLearning(vector<Example> examples, float alpha);
    friend void trainNeuralNet(vector<Example> examples, vector<Example> test, float alpha, float weightChangeThreshold, NeuralNet net);
};

void trainNeuralNet(vector<Example> examples, vector<Example> test, float alpha, float weightChangeThreshold, vector<float> hiddenLayerList, NeuralNet net);

#endif /* NeuralNet_hpp */
