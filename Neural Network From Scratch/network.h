#pragma once
#include "neuron.h"
#include <vector>
#include <utility>
#include "definitions.h"

using GradientTensor = std::vector<std::vector<std::pair<std::vector<dt>, dt>>>;

class NeuralNetwork {
public:
	enum Loss {
		MSE,
		CROSS_ENTROPY
	};

private:
	std::vector<std::vector<Neuron>> layers;
	Loss loss;
public:

	NeuralNetwork(const std::vector<int>& layerCount, const std::vector<Neuron::ActivationType>& types, Loss loss);

	std::vector<dt> forward(const std::vector<dt>& inputs);


	GradientTensor backward(const std::vector<dt>& outputs, const std::vector<dt>& trueOutpu);



	std::pair<dt, std::vector<int>> runBatch(const std::vector<std::vector<dt>>& inputs, const std::vector<std::vector<dt>>& trueOutput, dt learningRate);

	void gradientDescent(const std::vector<GradientTensor>& tensors, dt learningRate);

	std::tuple<GradientTensor, int, dt> run(const std::vector<dt>& inputs, const std::vector<dt>& trueOutput, dt learningRate);

	dt calculateLoss(const std::vector<dt>& outputs, const std::vector<dt>& trueOutput);

};
