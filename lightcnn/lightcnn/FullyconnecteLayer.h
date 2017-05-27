#pragma once
#include "config.h"
#include <iostream>
#include "SoftmaxLayer.h"
#include "BiasLayer.h"
class CFullyconnecteLayer
{
public:
	//全连接层的输入神经元个数，输出神经元个数
	CFullyconnecteLayer(int input_num,int output_num) {

		m_weights = Tensor2xf(input_num, output_num);
		m_dweights = Tensor2xf(input_num, output_num);
		m_bias = Tensor1xf(output_num);
		m_dbias = Tensor1xf(output_num);

	};
	~CFullyconnecteLayer() {


	};
	void CFullyconnecteLayer::test_set() {
		m_bias.setValues({ 3, 2 });
		Tensor1xf temp(3*2);
		temp.setValues({ 0.55f, 0.88f, 0.75f, 1.1f, 0.11f, 0.002f });

		m_weights = temp.reshape(Eigen::array<int, 2> { {3, 2}});

	}

	//y=x*w+b
	void CFullyconnecteLayer::forward(const Tensor2xf &bottom, Tensor2xf &top, const Eigen::ThreadPoolDevice &device) {

		top.device(device) = bottom.contract(m_weights, Eigen::array<Eigen::IndexPair<int>, 1> { Eigen::IndexPair<int>(1, 0) });
		auto b = m_bias.reshape(Eigen::array<int, 2>{ {1, top.dimension(1)}}).broadcast(Eigen::array<int, 2> { {bottom.dimension(0), 1}});
		top.device(device) = top+ b;
	}
	//http://cs231n.github.io/optimization-2/#mat
	//y=x*w+b,反向求导后dw=x.T*dy,dx=dy*w.T,db=dy
	void CFullyconnecteLayer::backward(const Tensor2xf&bottom,const Tensor2xf &d_top, Tensor2xf &d_bottom, const Eigen::ThreadPoolDevice &device) {

		Eigen::array<Eigen::IndexPair<int>, 1> product_dims_dw = { Eigen::IndexPair<int>(0, 0) };
		Eigen::array<Eigen::IndexPair<int>, 1> product_dims_di = { Eigen::IndexPair<int>(1, 1) };

		d_bottom.device(device) = d_top.contract(m_weights, product_dims_di);
		m_dweights.device(device) = bottom.contract(d_top, product_dims_dw);
		m_dbias.device(device) = d_top.sum(Eigen::array<int, 1> { { 0 } });
	}

public:

	//本层数据
	Tensor2xf m_weights;
	Tensor2xf m_dweights;
	Tensor1xf m_bias;
	Tensor1xf m_dbias;
};

class CFullyconnecteLayer_test
{
public:
	static void CFullyconnecteLayer_test::test() {

		Eigen::ThreadPool *tp = new Eigen::ThreadPool(8);
		Eigen::ThreadPoolDevice device(tp, 8);

		int batch_size = 4;
		int input_size = 3;
		int output_size = 2;
		Tensor2xf bottom(batch_size, input_size);
		bottom.setValues({ { 1,2,3 },{ 6,4,5 },{ 2,8,10 },{ 12,11,9 } });
		Tensor2xf top(batch_size, output_size);
		Tensor2xf dbottom(batch_size, input_size);
		Tensor2xf dtop(batch_size, output_size);

		Tensor1xf label_1d(batch_size);
		label_1d.setValues({ 1, 0, 1, 1 });
		Tensor2xf one_hot;
		CBaseFunction::onehot(label_1d, output_size, one_hot);


		CFullyconnecteLayer layer(input_size, output_size);
		layer.test_set();
		layer.forward(bottom, top,device);
		


		float loss = CBaseFunction::softmax_with_loss(top, one_hot, dtop, device);

		layer.backward(bottom, dtop, dbottom, device);




		






		std::cout << "***************backward************" << std::endl;
		std::cout << "loss" << loss << std::endl;
		std::cout << "m_top" << top << std::endl;
		std::cout << "m_dtop" << dtop << std::endl;
		std::cout << "m_dbottom" << dbottom << std::endl;
		std::cout << "m_dweight" << layer.m_dweights << std::endl;
		std::cout << "m_dbias" << layer.m_dbias << std::endl;



	}

};

