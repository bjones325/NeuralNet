//
//  Example.cpp
//  NeuralNet
//
//  Created by Blake Jones on 1/3/18.
//  Copyright © 2018 Blake Jones. All rights reserved.
//

#include "Example.hpp"

Example::Example(vector<float> input, vector<float> output) : inputList(input), outputList(output) {
    
}

vector<float> Example::getInputList() {
    return inputList;
}

vector<float> Example::getOutputList() {
    return outputList;
}

float Example::getInput(int index) {
    return inputList[index];
}

float Example::getOutput(int index) {
    return outputList[index];
}
