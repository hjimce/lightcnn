#pragma once
#include <Eigen/Eigen>
#include <iostream>
#include "SoftmaxLayer.h"
class CFullyconnecteLayer
{
public:
	CFullyconnecteLayer();
	~CFullyconnecteLayer();
	//y=x*w+b
	static void CFullyconnecteLayer::forward(const Eigen::MatrixXf &inputs, const Eigen::MatrixXf &weights, const Eigen::VectorXf &bais
		, Eigen::MatrixXf &outputs) {

		outputs = inputs*weights;
		outputs.rowwise() += bais.transpose();//每一行加上b
	}
	//y=x*w+b,反向求导后dw=x.T*dy,dx=dy*w.T,db=dy
	static void CFullyconnecteLayer::backward(const Eigen::MatrixXf &inputs, const Eigen::MatrixXf &weights, const Eigen::VectorXf &bais,
		const Eigen::MatrixXf &d_outputs,Eigen::MatrixXf &d_inputs, Eigen::MatrixXf &d_weights, Eigen::VectorXf &d_bais) {

		d_weights = inputs.transpose()*d_outputs;
		d_inputs = d_outputs*weights.transpose();
		d_bais = d_outputs.colwise().sum();
	}
	static void CFullyconnecteLayer::test() {//验证测试函数
		int batch_size = 4;
		int input_size = 3;
		int output_size = 2;
		Eigen::MatrixXf inputs(batch_size, input_size);
		inputs <<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
		Eigen::MatrixXf weights(input_size, output_size);
		weights <<  0.55, 0.88, 0.75, 1.1, 0.11, 0.002;
		Eigen::VectorXf bais(output_size);
		bais << 3, 2;
		Eigen::VectorXf label(batch_size);
		label << 1, 0, 1, 1;

		

		Eigen::MatrixXf outputs;//全连接层
		forward(inputs, weights, bais, outputs);
		
		Eigen::MatrixXf d_input, d_weights,d_output;
		float loss;
		CSoftmaxLayer::softmax_loss_forward_backward(outputs, label, d_output, loss);
		std::cout << loss << std::endl;

		Eigen::VectorXf d_bais;
		backward(inputs, weights, bais, d_output, d_input, d_weights, d_bais);


		std::cout << "outputs" << outputs << std::endl;
		std::cout << "d_output" << d_output << std::endl;
		std::cout << "d_input" << d_input << std::endl;
		std::cout << "d_weights" << d_weights << std::endl;
		std::cout << "d_bais"<<d_bais << std::endl;
	}
};

