//
//  main.cpp
//  NeuralNet
//
//  Created by Blake Jones on 12/30/17.
//  Copyright © 2017 Blake Jones. All rights reserved.
//

#include <iostream>
#include "Perceptron.hpp"

int main(int argc, const char * argv[]) {
    vector<int> items;
    Perceptron pc (1, items);
    pc.printInfo();
    vector<int> inputs({1});
    std::cout << "Sum: " << pc.getWeightedSum(inputs) <<endl;
    std::cout << "Hello, World!\n";
    return 0;
}
