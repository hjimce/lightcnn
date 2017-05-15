#pragma once
#include "config.h"
#include <iostream>
#include "SoftmaxLayer.h"

class CFullyconnecteLayer
{
public:

	//y=x*w+b
	static void CFullyconnecteLayer::forward(const Tensor2xf &inputs, const Tensor2xf &weights, const Tensor1xf &bais
		, Tensor2xf &outputs) {
		
		outputs = inputs.contract(weights, Eigen::array<Eigen::IndexPair<int>, 1> { Eigen::IndexPair<int>(1, 0) });

		auto b = bais.reshape(Eigen::array<int, 2>{ {1, outputs.dimension(1)}}).broadcast(Eigen::array<int, 2> { {inputs.dimension(0), 1}});

		
		outputs =outputs+b;
	//	outputs.rowwise() += bais

/*
		outputs = inputs*weights;
		outputs.rowwise() += bais.transpose();//每一行加上b*/
	}
   //http://cs231n.github.io/optimization-2/#mat
	//y=x*w+b,反向求导后dw=x.T*dy,dx=dy*w.T,db=dy
	static void CFullyconnecteLayer::backward(const Tensor2xf&inputs, const Tensor2xf &weights, const Tensor1xf &bais,
		const Tensor2xf &d_outputs, Tensor2xf &d_inputs, Tensor2xf &d_weights, Tensor1xf &d_bais) {

		Eigen::array<Eigen::IndexPair<int>, 1> product_dims_dw = { Eigen::IndexPair<int>(0, 0) };
		d_weights = inputs.contract(d_outputs, product_dims_dw);

		Eigen::array<Eigen::IndexPair<int>, 1> product_dims_di = { Eigen::IndexPair<int>(1, 1) };
		d_inputs = d_outputs.contract(weights, product_dims_di);


		d_bais = d_outputs.sum(Eigen::array<int, 1> { { 0 } });


/*
		d_weights = inputs.transpose()*d_outputs;
		d_inputs = d_outputs*weights.transpose();
		d_bais = d_outputs.colwise().sum();*/
	}
	static void CFullyconnecteLayer::test() {//验证测试函数

		int batch_size = 4;
		int input_size = 3;
		int output_size = 2;
		float input_data[12] = { 1, 2, 3, 6, 4, 5, 2, 8, 10, 12, 11, 9 };
		Eigen::TensorMap<Tensor2xf>inputs(input_data, batch_size, input_size);
		float weight_data[6] = { 0.55, 0.88, 0.75, 1.1, 0.11, 0.002 };
		Eigen::TensorMap<Tensor2xf>weights(weight_data, input_size, output_size);
		float bais_data[2] = { 3,2 };
		Eigen::TensorMap<Tensor1xf>bais(bais_data, output_size);
		float label_data[4] = { 1, 0, 1, 1 };
		Eigen::TensorMap<Tensor1xf> label(label_data, batch_size);


		

		Tensor2xf outputs;//全连接层
		forward(inputs, weights, bais, outputs);
		
		Tensor2xf d_input, d_weights,d_output;
		Tensor0xf loss;
		CSoftmaxLayer::softmax_loss_forward_backward(outputs, label, d_output, loss);
		std::cout << loss << std::endl;

		Tensor1xf d_bais;
		backward(inputs, weights, bais, d_output, d_input, d_weights, d_bais);


		std::cout << "outputs" << outputs << std::endl;
		std::cout << "d_output" << d_output << std::endl;
		std::cout << "d_input" << d_input << std::endl;
		std::cout << "d_weights" << d_weights << std::endl;
		std::cout << "d_bais"<<d_bais << std::endl;
	}
};

