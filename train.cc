#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <dirent.h>
#include <string>
#include <opencv2/opencv.hpp>

#if 0
#include <iostream>
#include <dirent.h>
#include <string>
#include <opencv2/opencv.hpp>  // 包含 OpenCV 头文件
class train_cls {
public:
	train_cls(char * folder) {
		this->folder = folder;	
	}
	int resize() {
		DIR* dir = opendir(this->folder);
		if (dir!= nullptr) {
			struct dirent * entry;
			while ((entry = readdir(dir))!= nullptr) {
				cv::Mat img = cv::imread(entry->d_name, cv::IMREAD_COLOR);
 				if (img.empty()) {
					std::cerr << "无法读取图片。" << std::endl;
				} else {
					std::cout << "read picture success" << std::endl;
				}
    			}
				
		}
		closedir(dir);
		return 0;
	} 	
	int train();
	int result();
private:
	char * folder;
};

int main(int argc, char **argv)
{
	int result_class;
	train_cls entity(argv[1]);
	//entity.resize(folder)
	//enttiy.train();
	//result_class = enttiy.result();
	return 0;
}
#endif

// 卷积核结构体，包含卷积核权重和尺寸信息
struct ConvolutionKernel {
    int kernelSize;
    std::vector<std::vector<double>> weights;
};

// 全连接层结构体，包含权重、偏置和输入输出维度信息
struct FullyConnectedLayer {
    int inputSize;
    int outputSize;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
};

// 激活函数，这里简单用ReLU作为示例
double ReLU(double x) {
    return std::max(0.0, x);
}

// 卷积操作函数
cv::Mat convolution(cv::Mat input, ConvolutionKernel kernel) {
    int inputHeight = input.rows;
    int inputWidth = input.cols;
    int inputChannels = input.channels();
    int kernelSize = kernel.kernelSize;
    int outputHeight = inputHeight - kernelSize + 1;
    int outputWidth = inputWidth - kernelSize + 1;
    cv::Mat output(outputHeight, outputWidth, CV_MAKETYPE(CV_64F, inputChannels));

    for (int y = 0; y < outputHeight; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            for (int c = 0; c < inputChannels; ++c) {
                double sum = 0;
                for (int ky = 0; ky < kernelSize; ++ky) {
                    for (int kx = 0; kx < kernelSize; ++kx) {
                        sum += input.at<cv::Vec3d>(y + ky, x + kx)[c] * kernel.weights[ky][kx];
                    }
                }
                output.at<cv::Vec3d>(y, x)[c] = sum;
            }
        }
    }

    return output;
}

// 全连接层前向传播函数
std::vector<double> fullyConnectedForward(std::vector<double> input, FullyConnectedLayer layer) {
    int inputSize = input.size();
    int outputSize = layer.outputSize;
    std::vector<double> output(outputSize, 0);
    for (int i = 0; i < outputSize; ++i) {
        double sum = 0;
        for (int j = 0; j < inputSize; ++j) {
            sum += input[j] * layer.weights[i][j];
        }
        output[i] = sum + layer.biases[i];
    }
    return output;
}

// 随机初始化卷积核权重
void initializeKernel(ConvolutionKernel& kernel) {
    int kernelSize = kernel.kernelSize;
    kernel.weights.resize(kernelSize, std::vector<double>(kernelSize));
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int y = 0; y < kernelSize; ++y) {
        for (int x = 0; x < kernelSize; ++x) {
            kernel.weights[y][x] = static_cast<double>(rand()) / RAND_MAX;  // 随机赋值在0到1之间
        }
    }
}

// 随机初始化全连接层权重和偏置
void initializeFullyConnectedLayer(FullyConnectedLayer& layer) {
    layer.weights.resize(layer.outputSize, std::vector<double>(layer.inputSize));
    layer.biases.resize(layer.outputSize);
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < layer.outputSize; ++i) {
        for (int j = 0; j < layer.inputSize; ++j) {
            layer.weights[i][j] = static_cast<double>(rand()) / RAND_MAX;  // 随机赋值在0到1之间
        }
        layer.biases[i] = static_cast<double>(rand()) / RAND_MAX;  // 随机赋值在0到1之间
    }
}

// 简单的训练函数，这里只是示意，实际训练逻辑更复杂
void train(cv::Mat inputImage, int label) {
    // 定义卷积层和全连接层示例
    ConvolutionKernel convKernel;
    convKernel.kernelSize = 3;
    initializeKernel(convKernel);

    FullyConnectedLayer fcLayer;
    fcLayer.inputSize = 638 * 638 * 3;  // 假设卷积后的尺寸，这里简化计算
    fcLayer.outputSize = 10;  // 假设输出10个类别
    initializeFullyConnectedLayer(fcLayer);

    // 前向传播
    cv::Mat convOutput = convolution(inputImage, convKernel);
    std::vector<double> flattenedInput;
    for (int y = 0; y < convOutput.rows; ++y) {
        for (int x = 0; x < convOutput.cols; ++x) {
            for (int c = 0; c < convOutput.channels(); ++c) {
                flattenedInput.push_back(convOutput.at<cv::Vec3d>(y, x)[c]);
            }
        }
    }
    std::vector<double> fcOutput = fullyConnectedForward(flattenedInput, fcLayer);

    // 这里简单打印输出，实际要根据输出和标签计算损失并更新权重（反向传播等）
    std::cout << "预测结果: ";
    for (double val : fcOutput) {
        std::cout << val << " ";
    }
    std::cout << " 真实标签: " << label << std::endl;
}

int main() {
    // 使用OpenCV读取图像，这里假设图像是640 * 640分辨率的RGB图像，路径根据实际情况修改
    cv::Mat inputImage = cv::imread("test.jpg", cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        std::cerr << "无法读取图像，请检查图像路径是否正确。" << std::endl;
        return -1;
    }
    // 将图像数据类型转换为CV_64F（因为之前代码中卷积等操作使用的是double类型数据）
    inputImage.convertTo(inputImage, CV_MAKETYPE(CV_64F, 3));

    // 假设训练标签示例（这里随意设为5）
    int label = 5;

    train(inputImage, label);

    return 0;
}
