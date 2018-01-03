//
//  Example.hpp
//  NeuralNet
//
//  Created by Blake Jones on 1/3/18.
//  Copyright © 2018 Blake Jones. All rights reserved.
//

#ifndef Example_hpp
#define Example_hpp

#include <stdio.h>
#include <vector>
#include <tuple>

using namespace std;

class Example {
private:
    vector<float> inputList;
    vector<float> outputList;
    
public:
    Example(vector<float> input, vector<float> output);
    ~Example();
    vector<float> getInputList();
    vector<float> getOutputList();
    float getInput(int index);
    float getOutput(int index);
};

#endif /* Example_hpp */
