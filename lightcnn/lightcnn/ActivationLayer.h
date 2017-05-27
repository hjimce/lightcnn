/*
#pragma once
#include "config.h"
#include "SoftmaxLayer.h"
#include "FullyconnecteLayer.h"*/
#pragma once
#include "config.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
enum ActivationMethod
{
	relu,
	sigmoid
};
template <class TType>
class CActivationLayer
{
public:
	CActivationLayer() {


	};
	~CActivationLayer() {


	};

	void CActivationLayer::forward(const TType &bottom,TType& top,
		const Eigen::ThreadPoolDevice &device, const ActivationMethod &method = ActivationMethod::relu) {
		switch (method)
		{
		case relu:
			top.device(device)= bottom.cwiseMax(0.f);
			break;
		case sigmoid:
			top.device(device)= 1.f / (1.f + (-bottom).exp());
			break;
		default:
			std::cout << "warning:activation method no support" << std::endl;
			break;
		}
	}

	void CActivationLayer::backward(const TType &dtop,const TType& top, TType&dbottom,
		const Eigen::ThreadPoolDevice &device, const ActivationMethod &method = ActivationMethod::relu) {
		switch (method)
		{
		case relu:
			dbottom.device(device) = (top> 0.f).cast<float>()*dtop;
			break;
		case sigmoid:
			dbottom.device(device) = dtop*top*(1.f - top);
			break;
		default:
			std::cout << "warning:activation backward method no support" << std::endl;
			break;
		}
	}



};
class CActivationLayer_test
{
public:
	static void CActivationLayer_test::test() {

		Eigen::ThreadPool *tp = new Eigen::ThreadPool(8);
		Eigen::ThreadPoolDevice device(tp, 8);



		CActivationLayer<Tensor3xf> layer;
		Tensor3xf bottom(2, 2, 3);
		Tensor3xf top(2, 2, 3);

		bottom.setRandom();
		layer.forward(bottom, top, device);


		std::cout << "***************forward************" << std::endl;



		Tensor3xf dtop(2, 2, 3);
		dtop.setConstant(2);
		Tensor3xf dbottom(2, 2, 3);
		layer.backward(dtop, top, dbottom, device);



		std::cout << "***************backward************" << std::endl;
		std::cout << "m_dtop"<<dtop << std::endl;
		std::cout << "m_dbottom"<<dbottom<< std::endl;




	}

};



/*
class CActivationLayer
{
	CActivationLayer(Eigen::ThreadPoolDevice *device) {
		m_device = device;

	};
	~CActivationLayer() {
	};
public:
	//relu=max(x,0)
	static void CActivationLayer::relu_forward(const Tensor2xf &bottom,Tensor2xf &top){
		
	}
	/ * if x>0 dx=drelu
	else dx=0  * /
	static void CActivationLayer::relu_backward(const Tensor2xf &top, const Tensor2xf &d_top, Tensor2xf &d_bottom) {
		
	}
	//y=1/(1+exp(-z))
	static void CActivationLayer::sigmoid_forward(const Tensor2xf &bottom, Tensor2xf &top) {
		
		
	}
	//dz=y(1-y)
	static void CActivationLayer::sigmoid_backward(const Tensor2xf &top, const Tensor2xf&d_top, Tensor2xf &d_bottom) {
		
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
		
};*/

