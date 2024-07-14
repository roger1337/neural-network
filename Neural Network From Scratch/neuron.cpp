#include <vector>
#include <random>
#include <iostream>
#include <cassert>
#include "neuron.h"
#include "definitions.h"
#include <chrono>

static std::default_random_engine generator(36435345);

Neuron::Neuron(int prevLayerCount, ActivationType type) : weights{ std::vector<dt>(prevLayerCount) }, activationType{ type }, value{} {
	initVals(prevLayerCount);
}

void Neuron::initVals(int prevLayerCount) {
	if (prevLayerCount > 0) {
		std::normal_distribution<dt> distribution(0, sqrt(2.0/prevLayerCount)); // distribution of mean 0 and standard deviation of 1
		for (int i{ 0 }; i < prevLayerCount; i++) {
			weights[i] = distribution(generator);
		}

		bias = distribution(generator);
	}
};

void Neuron::compute(const std::vector<Neuron>& prevLayer) {
	assert(prevLayer.size() == weights.size());
	dt sum{};

	for (int i{ 0 }; i < prevLayer.size(); i++) {
		sum += threshold(threshold(prevLayer[i].getActivation()) * threshold(weights[i]));
	}



	sum += threshold(bias);

	rawValue = sum;
	switch (activationType) {
	case SOFTMAX:
		// do nothing because we need to compute sums of logits first
		value = sum;
		break;
	case RELU:
		value = relU(sum);
		break;
	case SIGMOID:
		value = sigmoid(sum);
		break;
	}
}


