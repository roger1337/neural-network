#include <iostream>
#include "neuron.h"
#include "network.h"
#include "csv.hpp"
#include <vector>
#include <utility>
#include "definitions.h"


using namespace csv;

constexpr int BATCH_SIZE{ 50 };

int main() {
	std::vector<int> layerCount{ 784, 128, 10 };
	std::vector<Neuron::ActivationType> types{ Neuron::ActivationType::RELU, Neuron::ActivationType::RELU, Neuron::ActivationType::SOFTMAX };
	NeuralNetwork network{ layerCount, types, NeuralNetwork::CROSS_ENTROPY };

	for (int epoch{ 0 }; epoch < 3; epoch++) {
		std::cout << "Epoch: " << (epoch + 1) << '\n';
		CSVReader reader("C:\\mnist\\mnist_train.csv");

		std::vector<std::vector<dt>> batch;
		std::vector<std::vector<dt>> batchRealOutput;
		std::vector<int> realOutputNum;

		int num{ 0 };
		int batches{ 0 };


		for (int i{ 0 }; i < 60000; i++) {

			CSVRow& row{ *reader.begin() };

			int target{ row[0].get<int>() };
			//std::cout << "Target: " << target << '\n';

			std::vector<dt> realOutput(10);
			realOutput[target] = 1;

			std::vector<dt> input(row.size() - 1);
			for (int i{ 1 }; i < row.size(); i++) {
				input[i - 1] = (dt)row[i].get<int>() / 255;
			}

			batch.push_back(input);
			batchRealOutput.push_back(realOutput);
			realOutputNum.push_back(target);
			num++;

			if (num % BATCH_SIZE == 0) {
				batches++;

				std::pair<dt, std::vector<int>> res{ network.runBatch(batch, batchRealOutput, 0.08f) };

				int amtCorrect{};
				for (int i{ 0 }; i < BATCH_SIZE; i++) {
					if (res.second[i] == realOutputNum[i])
						amtCorrect++;
				}



				std::cout << "Batch " << batches << ": " << amtCorrect << "/ " << BATCH_SIZE << ". Average Loss: " << res.first << '\n';
				batch.clear();
				batchRealOutput.clear();
				realOutputNum.clear();
			}


		}
	}


	std::cout << "The test";
	int totalCorrect{};
	CSVReader reader("C:\\mnist\\mnist_test.csv");

	std::vector<std::vector<dt>> batch;
	std::vector<std::vector<dt>> batchRealOutput;
	std::vector<int> realOutputNum;

	int num{ 0 };
	int batches{ 0 };


	for (int i{ 0 }; i < 10000; i++) {

		CSVRow& row{ *reader.begin() };

		int target{ row[0].get<int>() };
		//std::cout << "Target: " << target << '\n';

		std::vector<dt> realOutput(10);
		realOutput[target] = 1;
			
		std::vector<dt> input(row.size() - 1);
		for (int i{ 1 }; i < row.size(); i++) {
			input[i - 1] = (dt)row[i].get<int>() / 255;
		}

		batch.push_back(input);
		batchRealOutput.push_back(realOutput);
		realOutputNum.push_back(target);
		num++;

		if (num % BATCH_SIZE == 0) {
			batches++;

			std::pair<dt, std::vector<int>> res{ network.runBatch(batch, batchRealOutput, 0.09f) };

			int amtCorrect{};
			for (int i{ 0 }; i < BATCH_SIZE; i++) {
				if (res.second[i] == realOutputNum[i])
					amtCorrect++;
			}

			totalCorrect += amtCorrect;



			std::cout << "Batch " << batches << ": " << amtCorrect << "/ " << BATCH_SIZE << ". Average Loss: " << res.first << '\n';
			batch.clear();
			batchRealOutput.clear();
			realOutputNum.clear();
		}


	}

	std::cout << "Total correct: " << totalCorrect << '\n';


	return 0;

} 