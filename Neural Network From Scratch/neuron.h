#pragma once
#include <vector>
#include <algorithm>
#include <iostream>
#include "definitions.h"

class Neuron {
public:
	enum ActivationType {
		SOFTMAX,
		RELU,
		SIGMOID
	};
private:
	std::vector<dt> weights;
	ActivationType activationType;
	dt bias;
	dt value;
	dt rawValue;

public:
	Neuron(int prevLayerCount, ActivationType type);

	dt getActivation() const { return value; };
	dt getRawValue() const { return rawValue; };
	std::vector<dt>& getWeights() {

		return weights;
	}

	dt& getBias() {
		return bias;
	}
	ActivationType getType() { return activationType; };
	void setActivation(dt v) { value = v; };
	void compute(const std::vector<Neuron>& prevLayer);

private:
	void initVals(int prevLayerCount);

	static dt relU(dt in) {
		return std::max(0.01L * in, in);
	}

	static dt sigmoid(dt in) {
		return in / (1 + exp(-in));
	}
};