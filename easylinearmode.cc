#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// 目标函数： y = wx + b
struct LinearModel {
	double w;  // 权重
	double b;  // 偏置

	LinearModel() : w(0.0), b(0.0) {}

	double predict(double x) {
		return w * x + b;
	}
};

// 均方误差损失函数
double compute_loss(const vector<double>& x, const vector<double>& y, LinearModel& model)
{
	double loss = 0.0;
	int n = x.size();
	for (int i = 0; i < n; i++) {
		double y_pred = model.predict(x[i]);
		loss += (y_pred - y[i]) * (y_pred - y[i]);  // (y_pred - y)^2
	}
	return loss / n;  // 均方误差
}

// 梯度下降优化
void train(LinearModel& model, const vector<double>& x, const vector<double>& y, double lr, int epochs)
{
	int n = x.size();
	for (int epoch = 0; epoch < epochs; epoch++) {
		double dL_dw = 0.0;
		double dL_db = 0.0;

		for (int i = 0; i < n; i++) {
			double y_pred = model.predict(x[i]);
			double error = y_pred - y[i];

			dL_dw += error * x[i];  // ∂L/∂w = (y_pred - y) * x
			dL_db += error;         // ∂L/∂b = (y_pred - y)
		}

		// 计算梯度的均值
		dL_dw /= n;
		dL_db /= n;

		// 更新参数 (梯度下降)
		model.w -= lr * dL_dw;
		model.b -= lr * dL_db;

		// 打印损失
		if (epoch % (epochs / 10) == 0) {
			cout << "Epoch " << epoch << " Loss: " << compute_loss(x, y, model) << endl;
		}
	}
}

int main()
{
	// 简单数据集 (线性关系: y = 2x + 1)
	vector<double> x = {1, 2, 3, 4, 5};
	vector<double> y = {3, 5, 7, 9, 11};  // 目标 y = 2x + 1

	LinearModel model;
	double learning_rate = 0.01;
	int epochs = 1000;

	train(model, x, y, learning_rate, epochs);

	cout << "Final model: y = " << model.w << "x + " << model.b << endl;
	return 0;
}
