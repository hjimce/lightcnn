#pragma once
#include "config.h"
#include "SoftmaxLayer.h"
#include "FullyconnecteLayer.h"
class CActivationLayer
{
public:
	//relu=max(x,0)
	static void CActivationLayer::relu_forward(const Tensor2xf &bottom,Tensor2xf &top){
		top = bottom.cwiseMax(0.f);
	}
	/* if x>0 dx=drelu
	else dx=0  */
	static void CActivationLayer::relu_backward(const Tensor2xf &top, const Tensor2xf &d_top, Tensor2xf &d_bottom) {
		d_bottom = (top > 0.f).cast<float>()*d_top;
	}
	//y=1/(1+exp(-z))
	static void CActivationLayer::sigmoid_forward(const Tensor2xf &bottom, Tensor2xf &top) {
		top = 1 / (1 + (-bottom).exp());
		
	}
	//dz=y(1-y)
	static void CActivationLayer::sigmoid_backward(const Tensor2xf &top, const Tensor2xf&d_top, Tensor2xf &d_bottom) {
		d_bottom = d_top*top*(1 - top);
	}
	static void CActivationLayer::test() {//验证测试函数

	int batch_size = 4;
	int input_size = 3;
	int output_size = 2;
	float input_data[12] = { 1, 2, 3, 6, 4, 5, 2, 8, 10, 12, 11, 9 };
	Eigen::TensorMap<Tensor2xf>inputs(input_data, batch_size, input_size);
	float weight_data[6] = { 0.55, -0.88, 0.75, -1.1, -0.11, 0.002 };
	Eigen::TensorMap<Tensor2xf>weights(weight_data, input_size, output_size);
	float bais_data[2] = { 3,-2 };
	Eigen::TensorMap<Tensor1xf>bais(bais_data, output_size);
	float label_data[4] = { 1, 0, 1, 1 };
	Eigen::TensorMap<Tensor1xf> label(label_data, batch_size);



	Tensor2xf fc_outputs;//全连接层-》forward
	CFullyconnecteLayer::forward(inputs, weights, bais, fc_outputs);
	Tensor2xf relu_output;//relu层->forward
	relu_forward(fc_outputs, relu_output);

	Tensor2xf d_input, d_fc_weights, d_relu_output,d_fc_output;//softmax层—forward_backward
	Tensor0xf loss;
	CSoftmaxLayer::softmax_loss_forward_backward(relu_output, label, d_relu_output, loss);

		
	relu_backward(relu_output, d_relu_output, d_fc_output);
		




	Tensor1xf d_fc_bais;
	CFullyconnecteLayer::backward(inputs, weights, bais, d_fc_output, d_input, d_fc_weights, d_fc_bais);

	std::cout << loss << std::endl;
	std::cout << "relu_outputs" << relu_output << std::endl;
	std::cout << "d_relu_output" << d_relu_output << std::endl;
	std::cout << "d_input" << d_input << std::endl;
	std::cout << "d_fc_weights" << d_fc_weights << std::endl;
	std::cout << "d_fc_bais" << d_fc_bais << std::endl;

		
	}
		
};

