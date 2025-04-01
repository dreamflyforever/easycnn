#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "ylog.h"
// 定义二维图像和三维张量数据结构
using Matrix = std::vector<std::vector<float>>;
using Tensor = std::vector<Matrix>;

// 初始化随机数种子
void initializeRandomSeed() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
}

// 随机生成值在 [-1, 1] 之间的浮点数
float randomFloat() {
    return static_cast<float>(std::rand()) / RAND_MAX * 2 - 1;
}

// 卷积层
class ConvLayer {
public:
    ConvLayer(int inChannels, int outChannels, int kernelSize)
        : inChannels(inChannels), outChannels(outChannels), kernelSize(kernelSize) {
        initializeWeights();
    }

    Tensor forward(const Tensor& input) {
        int outputSize = input[0].size() - kernelSize + 1;
        Tensor output(outChannels, Matrix(outputSize, std::vector<float>(outputSize, 0.0f)));

        for (int oc = 0; oc < outChannels; ++oc) {
            for (int ic = 0; ic < inChannels; ++ic) {
                for (int i = 0; i < outputSize; ++i) {
                    for (int j = 0; j < outputSize; ++j) {
                        float sum = 0.0f;
                        for (int ki = 0; ki < kernelSize; ++ki) {
                            for (int kj = 0; kj < kernelSize; ++kj) {
                                sum += input[ic][i + ki][j + kj] * weights[oc][ic][ki][kj];
                            }
                        }
                        output[oc][i][j] += sum;
                    }
                }
            }
        }
        return output;
    }

private:
    int inChannels, outChannels, kernelSize;
    std::vector<Tensor> weights;

    void initializeWeights() {
        weights.resize(outChannels, Tensor(inChannels, Matrix(kernelSize, std::vector<float>(kernelSize))));
        for (auto& oc : weights) {
            for (auto& ic : oc) {
                for (auto& row : ic) {
                    for (auto& val : row) {
                        val = randomFloat();
                    }
                }
            }
        }
    }
};

// 池化层 (最大池化)
class PoolLayer {
public:
    PoolLayer(int poolSize) : poolSize(poolSize) {}

    Tensor forward(const Tensor& input) {
        int outputSize = input[0].size() / poolSize;
        Tensor output(input.size(), Matrix(outputSize, std::vector<float>(outputSize, 0.0f)));

        for (int c = 0; c < input.size(); ++c) {
            for (int i = 0; i < outputSize; ++i) {
                for (int j = 0; j < outputSize; ++j) {
                    float maxVal = -1e9;
                    for (int pi = 0; pi < poolSize; ++pi) {
                        for (int pj = 0; pj < poolSize; ++pj) {
                            maxVal = std::max(maxVal, input[c][i * poolSize + pi][j * poolSize + pj]);
                        }
                    }
                    output[c][i][j] = maxVal;
                }
            }
        }
        return output;
    }

private:
    int poolSize;
};

// 全连接层
class FullyConnectedLayer {
public:
    FullyConnectedLayer(int inputSize, int outputSize)
        : inputSize(inputSize), outputSize(outputSize) {
        initializeWeights();
    }

    std::vector<float> forward(const std::vector<float>& input) {
        std::vector<float> output(outputSize, 0.0f);
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                output[i] += input[j] * weights[i][j];
            }
            output[i] = sigmoid(output[i]);
        }
        return output;
    }

    int inputSize, outputSize;
    std::vector<std::vector<float>> weights;
private:

    void initializeWeights() {
        weights.resize(outputSize, std::vector<float>(inputSize));
        for (auto& row : weights) {
            for (auto& val : row) {
                val = randomFloat();
            }
        }
    }

    float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
};

// 展平张量为向量
std::vector<float> flatten(const Tensor& tensor) {
    std::vector<float> flattened;
    for (const auto& matrix : tensor) {
        for (const auto& row : matrix) {
            for (float val : row) {
                flattened.push_back(val);
            }
        }
    }
    return flattened;
}

void disk_full(void * arg)
{
	ylog("no space\n");
}

float meanSquaredError(const std::vector<float>& predicted, const std::vector<float>& target) {
    float sum = 0.0f;
    for (size_t i = 0; i < predicted.size(); ++i) {
        float diff = predicted[i] - target[i];
        sum += diff * diff;
    }
    return sum / predicted.size();
}

class NeuralNetwork {
public:
    NeuralNetwork()
        : convLayer(3, 16, 3), poolLayer(2), fullyConnectedLayer(16 * 159 * 159, 10) {} // 初始化网络层，调整输出通道数和全连接层输入大小

    std::vector<float> forward(const Tensor& input) {
        auto convOutput = convLayer.forward(input);       // 卷积层
        auto pooledOutput = poolLayer.forward(convOutput); // 池化层
        auto flattened = flatten(pooledOutput);            // 展平
        return fullyConnectedLayer.forward(flattened);     // 全连接层
    }

    void train(const std::vector<Tensor>& inputs, const std::vector<std::vector<float>>& labels, 
               int epochs, float learningRate) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float totalLoss = 0.0f;
            for (size_t i = 0; i < inputs.size(); ++i) {
                // 前向传播
                auto output = forward(inputs[i]);

                // 计算损失
                float loss = meanSquaredError(output, labels[i]);
                totalLoss += loss;

                // 反向传播并更新权重（假设仅更新全连接层）
                for (size_t j = 0; j < output.size(); ++j) {
                    float error = output[j] - labels[i][j];
                    for (size_t k = 0; k < fullyConnectedLayer.inputSize; ++k) {
                        fullyConnectedLayer.weights[j][k] -= learningRate * error;
                    }
                }
            }
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Loss: " << totalLoss / inputs.size() << std::endl;
        }
    }

private:
    ConvLayer convLayer;
    PoolLayer poolLayer;
    FullyConnectedLayer fullyConnectedLayer;
};

int train_iter()
{
    // 初始化随机种子
    initializeRandomSeed();

    // 创建神经网络
    NeuralNetwork network;

    // 假设 inputs 和 labels 已准备好
    std::vector<Tensor> inputs;  // 训练输入数据
    std::vector<std::vector<float>> labels; // 训练标签，使用 one-hot 编码

    // 设置训练参数
    int epochs = 10;
    float learningRate = 0.01f;

    // 开始训练
    network.train(inputs, labels, epochs, learningRate);

    return 0;
}

/*测试网络结构*/
int main()
{
	ylog_init();
	register_cb(disk_full);
	ylog("compile time : %s\n", __TIME__);

	initializeRandomSeed();
	// 模拟 3 通道的 640x640 输入图像
	Tensor input(3, Matrix(640, std::vector<float>(640, 0.0f)));
	
	// 初始化输入数据（这里使用随机值作为示例）
	for (int c = 0; c < 3; ++c) {
		for (int i = 0; i < 640; ++i) {
			for (int j = 0; j < 640; ++j) {
				input[c][i][j] = randomFloat();
			}
		}
	}

	// 构建网络
	ConvLayer conv1(3, 16, 3);  // 3 输入通道，16 输出通道，3x3 卷积
	PoolLayer pool1(2);         // 2x2 池化
	FullyConnectedLayer fc1(16 * 159 * 159, 2);  // 全连接层，将展平后的向量映射到 2 类输出

	// 前向传播
	Tensor convOutput = conv1.forward(input);
	Tensor poolOutput = pool1.forward(convOutput);
	std::vector<float> flattened = flatten(poolOutput);
	std::vector<float> output = fc1.forward(flattened);

	// 输出结果
	std::cout << "网络输出: ";
	for (float val : output) {
		std::cout << val << " ";
	}
	std::cout << std::endl;

	ylog_deinit();
	return 0;
}
