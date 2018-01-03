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

class NeuralNet {
private:
    int layerSize;
    int numberHiddenLayers;
    int numberLayers;
    vector<Perceptron> outputLayer;
    vector<vector<Perceptron>> hiddenLayers;
    
public:
    NeuralNet(int layerSize);
    ~NeuralNet();
    vector<vector<Perceptron>> feedForward(vector<float> &inActuals);
    tuple<int> backwardPropLearning(int examples, int alpha);
    
};

#endif /* NeuralNet_hpp */
