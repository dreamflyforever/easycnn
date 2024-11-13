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

private:
    int inputSize, outputSize;
    std::vector<std::vector<float>> weights;

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

/*测试网络结构*/
int main()
{
	ylog_init();
	register_cb(disk_full);
	ylog("compile time : %s\n", __TIME__);

	initializeRandomSeed();
	ylog("");
	// 模拟 1 通道的 8x8 输入图像
	Tensor input = {{{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}}};

	// 构建网络
	ConvLayer conv1(1, 2, 3);  // 1 输入通道，2 输出通道，3x3 卷积
	PoolLayer pool1(2);        // 2x2 池化
	FullyConnectedLayer fc1(8, 2);  // 全连接层，将展平后的向量映射到 2 类输出

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
