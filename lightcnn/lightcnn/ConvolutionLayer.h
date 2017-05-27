#pragma once
#include "config.h"
#include <iostream>
#include <string.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include "BaseFunction.h"

class CConvolutionLayer
{
public:
	//bottom_shape=([batch, in_height, in_width, in_channels])
	//weights_shape=(filter_height, filter_width, in_channels, out_channels)
	CConvolutionLayer(std::vector<int>input_shape,std::vector<int>kenel_shape,std::vector<int>stride_shape,PaddingMethod convolution_method
	) {
		m_weights = Tensor4xf(kenel_shape[0], kenel_shape[1],kenel_shape[2],kenel_shape[3]);
		m_dweights = Tensor4xf(m_weights);
		m_bias = Tensor1xf(kenel_shape[3]);
		m_dbias = Tensor1xf(kenel_shape[3]);
		m_convolution_method = convolution_method;
	};
	~CConvolutionLayer() {


	};
	//bottom_shape=([batch, in_height, in_width, in_channels])
	//weights_shape=(filter_height, filter_width, in_channels, out_channels)
	//采用valid 卷积
	void CConvolutionLayer::valid_convolution(const Tensor4xf &bottom, const Tensor4xf&kenel, Tensor4xf & top) {
		top.setZero();
		for (int ob = 0; ob < top.dimension(0); ++ob) {
			for (int oc = 0; oc < top.dimension(3); ++oc) {
				for (int oh = 0; oh < top.dimension(1); ++oh) {
					for (int ow = 0; ow < top.dimension(2); ++ow) {
						auto &result = top(ob, oh, ow, oc);
						for (int ic = 0; ic < bottom.dimension(3); ic++)
						{
							for (int kh = 0; kh < kenel.dimension(0); ++kh)
							{
								for (int kw = 0; kw < kenel.dimension(1); ++kw)
								{
									result += bottom(ob, oh + kh, ow + kw, ic) * kenel(kh, kw, ic, oc);
								}
							}
						}
					}
				}
			}
		}
	}
	void CConvolutionLayer::forward(const Tensor4xf &bottom, Tensor4xf &top, const Eigen::ThreadPoolDevice &device) {
		switch (m_convolution_method)
		{
		case valid:
			valid_convolution(bottom, m_weights, top);
			break;
		case same:
			break;
		default:
			break;
		}
		//CBaseFunction::add_bias(top, m_bias, top,device);

	}
	//bottom_shape=([batch, in_height, in_width, in_channels])
	//weights_shape=(filter_height, filter_width, in_channels, out_channels)
	static void CConvolutionLayer::backward(const Tensor4xf &bottom, const Tensor4xf&weights, const Tensor1xf&bais,const Tensor4xf &d_top,
		Tensor4xf &d_bottom,Tensor4xf &d_weights,Tensor1xf &d_bais) {
		d_bais= d_top.sum(Eigen::array<int, 3> { 0,1,2});
		auto flipw= weights.reverse(Eigen::array<bool, 4> { false, true, false,  false });
		
	}
public:
	Tensor4xf m_weights;
	Tensor4xf m_dweights;
	Tensor1xf m_bias;
	Tensor1xf m_dbias;
	PaddingMethod m_convolution_method;
};


class CConvolutionLayer_test
{
public:
	static void CConvolutionLayer_test::test() {

		Eigen::ThreadPool *tp = new Eigen::ThreadPool(8);
		Eigen::ThreadPoolDevice device(tp, 8);



		int batch_size = 3;
		int input_channel = 2;
		int input_height = 5;
		int input_width = 5;
		int output_channel = 2;
		int kenel_height = 3;
		int kenel_widht = 3;
		float *input_data = new float[batch_size*input_channel*input_height*input_width];
		for (int i = 0; i<batch_size*input_channel*input_height*input_width; i++)
		{
			input_data[i] = 0.1*i;
		}
		Eigen::TensorMap<Tensor4xf>inputs(input_data, batch_size, input_height, input_width, input_channel);

		float *weight_data = new float[input_channel*output_channel*kenel_height*kenel_widht];
		for (int i = 0; i<input_channel*output_channel*kenel_height*kenel_widht; i++)
		{
			weight_data[i] = 1.f / float(1 + i);
		}
		Eigen::TensorMap<Tensor4xf> weights(weight_data, kenel_height, kenel_widht, input_channel, output_channel);

		float *bais_data = new float[output_channel];
		for (int i = 0; i < output_channel; i++)
		{
			bais_data[i] = i*0.2;
		}
		Eigen::TensorMap<Tensor1xf> bais(bais_data, output_channel);

		Tensor4xf conv1(inputs.dimension(0), inputs.dimension(1) - weights.dimension(0) + 1,
			inputs.dimension(2) - weights.dimension(1) + 1, weights.dimension(3));

		CConvolutionLayer layer({ batch_size,input_height,input_width,input_channel },
		{ kenel_height,kenel_widht,input_channel,output_channel }, { 2 },PaddingMethod::valid);
		layer.m_weights = weights;
		layer.m_bias = bais;
		layer.forward(inputs, conv1,device);



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

