#pragma once
#include "config.h"
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
class CConvolutionLayer
{
public:
	//bottom_shape=([batch, in_height, in_width, in_channels])
	//weights_shape=(filter_height, filter_width, in_channels, out_channels)
	//采用valid 卷积
	static void CConvolutionLayer::forward(const Tensor4xf &bottom,const Tensor4xf&weights,
		const Tensor1xf&bais, Tensor4xf &top) {
		top.resize(bottom.dimension(0), bottom.dimension(1)-weights.dimension(0)+1, 
			bottom.dimension(2)-weights.dimension(1)+1, weights.dimension(3));
		top.setZero();


		for (int ob = 0; ob < top.dimension(0); ++ob) {
			for (int oc = 0; oc< top.dimension(3); ++oc) {
				for (int oh = 0; oh< top.dimension(1); ++oh) {
					for (int ow = 0; ow < top.dimension(2); ++ow) {
						auto &result = top(ob, oh, ow, oc);
						for (int ic=0;ic<bottom.dimension(3);ic++)
						{
							for (int kh=0;kh<weights.dimension(0);++kh)
							{
								for (int kw=0;kw<weights.dimension(1);++kw)
								{
									result += bottom(ob, oh+kh, ow+ kw, ic) * weights(kh, kw, ic, oc);
								}
							}
						}
						result += bais(oc);
					}
				}
			}
		}


	}
	static void CConvolutionLayer::backward(const Tensor4xf &bottom, const Tensor4xf&weights, const Tensor1xf&bais,const Tensor4xf &d_top) {


	}
	static void CConvolutionLayer::test() {//验证测试函数
		int batch_size = 3;
		int input_channel = 2;
		int input_height = 5;
		int input_width = 5;
		int output_channel = 2;
		int kenel_height = 3;
		int kenel_widht = 3;
		float *input_data = new float[batch_size*input_channel*input_height*input_width];
		for (int i=0;i<batch_size*input_channel*input_height*input_width;i++)
		{
			input_data[i] = 0.1*i;
		}
		Eigen::TensorMap<Tensor4xf>inputs(input_data,batch_size,input_height,input_width,input_channel);

		float *weight_data = new float[input_channel*output_channel*kenel_height*kenel_widht];
		for (int i=0;i<input_channel*output_channel*kenel_height*kenel_widht;i++)
		{
			weight_data[i] = 1.f/ float(1 + i);
		}
		Eigen::TensorMap<Tensor4xf> weights(weight_data,kenel_height,kenel_widht,input_channel, output_channel);

		float *bais_data = new float[output_channel];
		for (int i = 0; i < output_channel; i++)
		{
			bais_data[i] =i*0.2;
		}
		Eigen::TensorMap<Tensor1xf> bais(bais_data, output_channel);

		Tensor4xf conv1;
		forward(inputs, weights, bais, conv1);

		


		//std::cout << "inputs" << inputs << std::endl;
		
		//std::cout << "top shape:" << top.dimension(0)<< top.dimension(1) << top.dimension(2) << top.dimension(3) << std::endl;
		std::cout << "top shape:" << conv1 << std::endl;


		/*
		weights << 0.55, 0.88, 0.75, 1.1, 0.11, 0.002;
		Eigen::VectorXf bais(output_size);
		bais << 3, 2;
		Eigen::VectorXf label(batch_size);
		label << 1, 0, 1, 1;*/



		/*Eigen::MatrixXf outputs;//全连接层
		forward(inputs, weights, bais, outputs);

		Eigen::MatrixXf d_input, d_weights, d_output;
		float loss;
		CSoftmaxLayer::softmax_loss_forward_backward(outputs, label, d_output, loss);
		std::cout << loss << std::endl;

		Eigen::VectorXf d_bais;
		backward(inputs, weights, bais, d_output, d_input, d_weights, d_bais);


		std::cout << "outputs" << outputs << std::endl;
		std::cout << "d_output" << d_output << std::endl;
		std::cout << "d_input" << d_input << std::endl;
		std::cout << "d_weights" << d_weights << std::endl;
		std::cout << "d_bais" << d_bais << std::endl;*/
	}
};

