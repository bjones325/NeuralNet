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
    tuple<float> backPropLearning(vector<Example> examples, float alpha);
};

#endif /* NeuralNet_hpp */
