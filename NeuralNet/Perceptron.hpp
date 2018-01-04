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
    vector<float> weights;
    
public:
    Perceptron(int inSize, vector<float> &inWeights);
    
    ~Perceptron();
    
    float getWeightedSum(vector<float> &inActuals);
    
    float sigmoidActivation(vector<float> &inActuals);
    
    float sigmoidActivationDeriv(vector<float> &inActuals);
    
    float updateWeights(vector<float> &inActuals, float alpha, float delta);
    
    void setRandomWeights();
    
    vector<float> getWeights();
    
    int getInputSize();
    
    void printInfo();
};

#endif /* Perceptron_hpp */
