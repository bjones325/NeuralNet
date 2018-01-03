//
//  Perceptron.hpp
//  NeuralNet
//
//  Created by Blake Jones on 12/30/17.
//  Copyright © 2017 Blake Jones. All rights reserved.
//

#ifndef Perceptron_hpp
#define Perceptron_hpp

#include <stdio.h>
#include <iostream>
#include "math.h"
#include <vector>

using namespace std;

float sigmoid(float value);
float sigmoidDeriv(float value);

class Perceptron {
    
private:
    int inputSize = 1;
    vector<int> weights;
    
public:
    Perceptron(int inSize, vector<int> inWeights);
    
    ~Perceptron();
    
    float getWeightedSum(vector<int> &inActuals);
    
    float sigmoidActivation(vector<int> &inActuals);
    
    float sigmoidActivationDeriv(vector<int> &inActuals);
    
    float updateWeights(vector<int> &inActuals, float alpha, float delta);
    
    void setRandomWeights();
    
    void printInfo();
};

#endif /* Perceptron_hpp */
