#pragma once
#include <Eigen/Eigen>
#include "SoftmaxLayer.h"
#include "FullyconnecteLayer.h"
class CActivationLayer
{
public:
	CActivationLayer();
	~CActivationLayer();
	//relu=max(x,0)
	static void CActivationLayer::relu_forward(const Eigen::MatrixXf &inputs,Eigen::MatrixXf &outputs){
		outputs = inputs.cwiseMax(0);
	}
	/* if x>0 dx=drelu
	else dx=0  */
	static void CActivationLayer::relu_backward(const Eigen::MatrixXf &output , const Eigen::MatrixXf &d_outputs,Eigen::MatrixXf &d_inputs) {
		d_inputs = (output.array() <=0).select(0, d_outputs);
	}
	//y=1/(1+exp(-z))
	static void CActivationLayer::sigmoid_forward(const Eigen::MatrixXf &inputs, Eigen::MatrixXf &outputs) {
		outputs = 1 / (1 + (-inputs).array().exp());
		
	}
	//dz=y(1-y)
	static void CActivationLayer::sigmoid_backward(const Eigen::MatrixXf &output, const Eigen::MatrixXf &d_outputs, Eigen::MatrixXf &d_inputs) {
		d_inputs=


	}
	static void CActivationLayer::test() {//验证测试函数
		int batch_size = 4;
		int input_size = 3;
		int output_size = 2;
		Eigen::MatrixXf inputs(batch_size, input_size);
		inputs << 1, -2, 3, 4, -5, 6, 7, -8, -9, 10, 11, -12;
		Eigen::MatrixXf weights(input_size, output_size);
		weights << 0.55,-0.88, 0.75, -1.1, -0.11, 0.002;
		Eigen::VectorXf bais(output_size);
		bais << 3, -2;
		Eigen::VectorXf label(batch_size);
		label << 1, 0, 1, 1;



		Eigen::MatrixXf fc_outputs;//全连接层-》forward
		CFullyconnecteLayer::forward(inputs, weights, bais, fc_outputs);
		Eigen::MatrixXf relu_output;//relu层->forward
		relu_forward(fc_outputs, relu_output);

		Eigen::MatrixXf d_input, d_fc_weights, d_relu_output,d_fc_output;//softmax层—forward_backward
		float loss;
		CSoftmaxLayer::softmax_loss_forward_backward(relu_output, label, d_relu_output, loss);

		
		relu_backward(relu_output, d_relu_output, d_fc_output);
		




		Eigen::VectorXf d_fc_bais;
		CFullyconnecteLayer::backward(inputs, weights, bais, d_fc_output, d_input, d_fc_weights, d_fc_bais);

		std::cout << loss << std::endl;
		std::cout << "relu_outputs" << relu_output << std::endl;
		std::cout << "d_relu_output" << d_relu_output << std::endl;
		std::cout << "d_input" << d_input << std::endl;
		std::cout << "d_fc_weights" << d_fc_weights << std::endl;
		std::cout << "d_fc_bais" << d_fc_bais << std::endl;
		std::cout << "d_fc_output" << d_fc_output << std::endl;
		
	}
		
};

