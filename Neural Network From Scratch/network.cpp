#include "network.h"
#include <iostream>
#include <cassert>
#include "definitions.h"

constexpr dt GRADIENT_CLIP = 500.0L;
NeuralNetwork::NeuralNetwork(const std::vector<int>& layerCount, const std::vector<Neuron::ActivationType>& types, NeuralNetwork::Loss l) : layers{ std::vector<std::vector<Neuron>>(layerCount.size()) }, loss{ l } {
	assert(layerCount.size() > 1);
	assert(layerCount.size() == types.size());

	for (int i{ 0 }; i < layerCount.size(); i++) {
		layers[i] = std::vector<Neuron>{};
		layers[i].reserve(layerCount[i]);
		for (int j{ 0 }; j < layerCount[i]; j++) {
			layers[i].push_back(Neuron{ i == 0 ? 0 : layerCount[i - 1], types[i] });
		}
	}
}

// returns average loss, and also the most activated output neuron indices
std::pair<dt, std::vector<int>> NeuralNetwork::runBatch(const std::vector<std::vector<dt>>& inputs, const std::vector<std::vector<dt>>& trueOutputs, dt learningRate) {
	assert(inputs.size() == trueOutputs.size());

	std::vector<GradientTensor> gradientTensors(inputs.size());
	std::vector<int> mostActivatedIdx(inputs.size());

	dt averageLoss{};

	for (int i{ 0 }; i < inputs.size(); i++) {
		std::tuple<GradientTensor, int, dt> res{ run(inputs[i], trueOutputs[i], learningRate) };
		gradientTensors[i] = std::move(std::get<0>(res));
		averageLoss += std::get<2>(res);
		mostActivatedIdx[i] = std::get<1>(res);
	}

	averageLoss /= inputs.size();

	gradientDescent(gradientTensors, learningRate);

	return { averageLoss, mostActivatedIdx };
}

void NeuralNetwork::gradientDescent(const std::vector<GradientTensor>& tensors, dt learningRate) {
	for (int i{ 0 }; i < tensors[0].size(); i++) {
		for (int j{ 0 }; j < tensors[0][i].size(); j++) {
			for (int k{ 0 }; k < tensors[0][i][j].first.size(); k++) {

				dt averageGradient{};
				for (int a{ 0 }; a < tensors.size(); a++) {
					averageGradient += tensors[a][i][j].first[k];
				}
				averageGradient /= tensors.size();

				layers[i][j].getWeights()[k] += (-averageGradient) * learningRate;
			}

			dt averageBiasGradient{};

			for (int a{ 0 }; a < tensors.size(); a++) {
				averageBiasGradient += tensors[a][i][j].second;
			}
			averageBiasGradient /= tensors.size();

			layers[i][j].getBias() += (-averageBiasGradient) * learningRate;
		}
	}
}


std::tuple<GradientTensor, int, dt> NeuralNetwork::run(const std::vector<dt>& inputs, const std::vector<dt>& trueOutput, dt learningRate) {
	std::vector<dt> outputs{ forward(inputs) };

	dt loss{ calculateLoss(outputs, trueOutput) };

	GradientTensor result{ backward(outputs, trueOutput) };

	dt max{};
	int idx{ 0 };
	for (int i{ 0 }; i < outputs.size(); i++) {
		if (outputs[i] >= max) {
			max = outputs[i];
			idx = i;
		}
	}

	return std::tuple{ result, idx, loss };
}

dt NeuralNetwork::calculateLoss(const std::vector<dt>& outputs, const std::vector<dt>& trueOutput) {
	assert(outputs.size() == trueOutput.size());

	dt l{};

	switch(loss) {
	case CROSS_ENTROPY:
		for (int i{ 0 }; i < outputs.size(); i++) {
			// make sure this log is correct
			l -= trueOutput[i] * log(outputs[i] + 1e-50);
		}
		break;
	case MSE:
		for (int i{ 0 }; i < outputs.size(); i++) {
			l += (trueOutput[i] - outputs[i]) * (trueOutput[i] - outputs[i]);
		}
		l /= outputs.size();
		break;
	}

	return l;
}


GradientTensor NeuralNetwork::backward(const std::vector<dt>& outputs, const std::vector<dt>& trueOutput) {

	std::vector<dt> preActivationDerivatives(outputs.size());
	switch (loss) {
	case CROSS_ENTROPY: // assume we are using softmax. kinda bad solution but would require large restructure
		for (int i{ 0 }; i < outputs.size(); i++) {
			preActivationDerivatives[i] = threshold(outputs[i] - trueOutput[i]);
		}
		break;
	case MSE:
		for (int i{ 0 }; i < outputs.size(); i++) {
			dt f{ 2 * threshold(outputs[i] - trueOutput[i])};

			switch (layers[layers.size() - 1][i].getType()) {
			case Neuron::ActivationType::RELU:
				preActivationDerivatives[i] = layers[layers.size() - 1][i].getRawValue() <= 0 ? 0.01 * f : f;
				break;
			}
		}
		break;
	}

	// layers, then neurons, then a pair of list of weight gradients and bias gradient
	GradientTensor gradients(layers.size());


	for (size_t i{ layers.size() - 1 }; i >= 1; i--) {


		std::vector<dt> next(layers[i - 1].size()); // next layers neuron pre activation derivatives
		std::vector<std::pair<std::vector<dt>, dt>> layerGrads(layers[i].size());

		for (int j{ 0 }; j < layers[i].size(); j++) {
			Neuron& currentNeuron{ layers[i][j] };

			std::pair<std::vector<dt>, dt> grads;
			grads.first = std::vector<dt>(layers[i - 1].size());

			for (int k{ 0 }; k < grads.first.size(); k++) {
				Neuron& neuronConnectedToWeight{ layers[i - 1][k] };

				grads.first[k] = std::min(neuronConnectedToWeight.getActivation() * preActivationDerivatives[j], GRADIENT_CLIP);
		
				// we plus the gradient
				next[k] += currentNeuron.getWeights()[k] * preActivationDerivatives[j];

	
			}

			grads.second = preActivationDerivatives[j];
			layerGrads[j]= std::move(grads);
		}


		gradients[i] = std::move(layerGrads);

		for (int j{ 0 }; j < next.size(); j++) {
			switch (layers[i - 1][j].getType()) {
			case Neuron::ActivationType::RELU:

				next[j] = std::min(layers[i - 1][j].getRawValue() <= 0 ? 0.01 * next[j] : next[j], GRADIENT_CLIP);
				break;


			}
		}

		swap(preActivationDerivatives, next);
	}

	return gradients;
}



std::vector<dt> NeuralNetwork::forward(const std::vector<dt>& inputs) {
	assert(inputs.size() == layers[0].size());

	for (int i{ 0 }; i < layers[0].size(); i++) {
		layers[0][i].setActivation(inputs[i]);
	}

	for (int i{ 1 }; i < layers.size(); i++) {
		for (int j{ 0 }; j < layers[i].size(); j++) {
			layers[i][j].compute(layers[i - 1]);
		}
	}
	std::vector<dt> outputs(layers[layers.size() - 1].size());

	dt maxOutput{};
	for (int i{ 0 }; i < outputs.size(); i++) {
		maxOutput = std::max(layers[layers.size() - 1][i].getActivation(), maxOutput);
	}


	dt softmaxSum{};
	for (int i{ 0 }; i < layers[layers.size() - 1].size(); i++) {

		if (layers[layers.size() - 1][i].getType() == Neuron::ActivationType::SOFTMAX) {
			softmaxSum += threshold(exp(layers[layers.size() - 1][i].getActivation()-maxOutput));
			continue;
		}

		outputs[i] = layers[layers.size() - 1][i].getActivation();

	}

	
	for (int i{ 0 }; i < layers[layers.size() - 1].size(); i++) {
		if (layers[layers.size() - 1][i].getType() == Neuron::ActivationType::SOFTMAX) {
			outputs[i] = threshold(exp(layers[layers.size() - 1][i].getActivation()- maxOutput)) / softmaxSum;
		}

	}


	return outputs;
}
