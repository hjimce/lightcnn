#pragma once
#include <Eigen/Eigen>
#include <iostream>
class CSoftmaxLayer
{
public:
	CSoftmaxLayer();
	~CSoftmaxLayer();
	static void CSoftmaxLayer::softmax_function(const Eigen::MatrixXf &inputs, Eigen::MatrixXf &softmax) {
		softmax = inputs.array().exp();
		Eigen::VectorXf sorfmax_rowsum = softmax.rowwise().sum();
		softmax = softmax.array().colwise() / sorfmax_rowsum.array();//行归一化
	}
	//假设前一层网络经过全连接input=wx+b后，经过soft损失函数求导：sotfmax_net-pro_real
	//https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
	static void CSoftmaxLayer::softmax_loss_forward_backward(const Eigen::MatrixXf &inputs, const Eigen::VectorXf &label, Eigen::MatrixXf  &d_inputs, float &loss) {

		Eigen::MatrixXf softmax;
		softmax_function(inputs, softmax);


		Eigen::MatrixXf real_label = Eigen::MatrixXf::Zero(softmax.rows(), softmax.cols());
		assert(label.rows() == inputs.rows());
		for (int i = 0; i < label.rows(); i++)
		{
			real_label(i, label(i)) = 1;
		}

		loss = -(real_label.array()*softmax.array().log()).mean();//交叉熵损失函数平均值，平均值反向求导的时候，需要记住梯度除以
		d_inputs = (softmax - real_label) / (inputs.rows()*inputs.cols());//由于loss计算的时候，我们一般是计算loss mean，所以反向求导的时候，需要除以(inputs.rows()*inputs.cols())

	}
	static void CSoftmaxLayer::test() {
		int batch_size = 4;
		int input_size = 3;
		int output_size = 2;
		Eigen::MatrixXf inputs(batch_size, input_size);
		inputs << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
		Eigen::VectorXf label(batch_size);
		label << 1, 0, 1, 1;


		Eigen::MatrixXf  d_inputs;
		float loss;
		CSoftmaxLayer::softmax_loss_forward_backward(inputs, label, d_inputs, loss);
		std::cout << loss << std::endl;
		std::cout << d_inputs << std::endl;
	}
};

