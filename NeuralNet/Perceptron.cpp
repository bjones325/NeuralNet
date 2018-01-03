//
//  Perceptron.cpp
//  NeuralNet
//
//  Created by Blake Jones on 12/30/17.
//  Copyright © 2017 Blake Jones. All rights reserved.
//

#include "Perceptron.hpp"

float Perceptron::getWeightedSum(vector<float> &inActuals) {
    float sum = 0;
    for (int i = 0; i < inActuals.size(); i++) {
        sum += inActuals[i] * weights[i];
    }
    return sum;
}

float Perceptron::sigmoidActivation(vector<float> &inActuals) {
    vector<float> biasedActuals = inActuals;
    biasedActuals.push_back(1.0);
    float sum = getWeightedSum(biasedActuals);
    return sigmoid(sum);
}

float Perceptron::sigmoidActivationDeriv(vector<float> &inActuals) {
    vector<float> biasedActuals = inActuals;
    biasedActuals.push_back(1.0);
    float sum = getWeightedSum(biasedActuals);
    return sigmoidDeriv(sum);
}

float Perceptron::updateWeights(vector<float> &inActuals, float alpha, float delta) {
    float modification = 0.0;
    vector<float> biasedActuals = inActuals;
    biasedActuals.push_back(1.0);
    
    for (int i = 0; i < biasedActuals.size(); i++) {
        float change = alpha * biasedActuals[i] * delta;
        weights[i] = weights[i] + change;
        modification += abs(change);
    }
    return modification;
}

void Perceptron::setRandomWeights() {
    srand((unsigned int) time(NULL));
    for (int i = 0; i < inputSize - 1; i++) {
        weights[i] = (random() + 0.001) * ((random() % 2) == 1 ? 1 : -1);
    }
}

void Perceptron::printInfo() {
    std::cout << "InputSize " << inputSize << endl;
    std::cout << "Weights:" << endl;
    
    for (int i = 0; i < inputSize - 1; i++) {
        std::cout << "   " << weights[i] << endl;
    }
}

Perceptron::Perceptron(int inSize, vector<int> &inWeights) : inputSize(inSize + 1) {
    if (inWeights.size() == 0) {
        weights = inWeights;
    } else {
        /*for (int i = 0; i < inSize; i++) {
            weights[i] = 1;
        }*/
        setRandomWeights();
    }
}

Perceptron::~Perceptron() {
    
}

float sigmoid(float value) {
    return 1/(1 + exp(value));
}

float sigmoidDeriv(float value) {
    return sigmoid(value) * (1-sigmoid(value));
}
