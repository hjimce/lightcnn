#pragma once

#include "config.h"
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include "BaseFunction.h"

class CSoftmaxLayer_test
{
public:
	static void CSoftmaxLayer_test::test() {

		Eigen::ThreadPool *tp = new Eigen::ThreadPool(8);
		Eigen::ThreadPoolDevice device(tp, 8);



		int batch_size = 4;
		int input_size = 3;
		Tensor2xf inputs(batch_size, input_size);
		inputs.setValues({ {1,2,3},{6,5,4},{2,8,10},{12,11,9} });

		Tensor1xf label_1d(batch_size);
		label_1d.setValues({ 1, 0, 1, 1 });
		Tensor2xf one_hot;
		CBaseFunction::onehot(label_1d, 3,one_hot);

		Tensor2xf dinputs(batch_size, input_size);
		float loss = CBaseFunction::softmax_with_loss(inputs, one_hot, dinputs, device);



	

		std::cout << loss << std::endl;
		std::cout << dinputs << std::endl;






	}

};






/*
class CSoftmaxLayer
{
public:
	
	static void CSoftmaxLayer::softmax_function(const TType &inputs, TType &softmax) {
		softmax = inputs.exp();
		TType sorfmax_rowsum = softmax.sum(Eigen::array<int, 1>{ {1}}).reshape(Eigen::array<int, 2>{ {inputs.dimension(0), 1}});//第一维降维,这边不能用auto，会出现计算数值错误
		auto sorfmax_rowsum_broad = sorfmax_rowsum.broadcast(Eigen::array<int, 2> { {1, inputs.dimension(1)}});
		
		softmax = softmax/ sorfmax_rowsum_broad;//行归一化
	}
	//假设前一层网络经过全连接input=wx+b后，经过soft损失函数求导：sotfmax_net-pro_real
	//https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
	static void CSoftmaxLayer::softmax_loss_forward_backward(const TType &inputs, const Tensor1xf&label, TType &d_inputs, Tensor0xf &loss) {

		TType softmax;
		softmax_function(inputs, softmax);

		TType real_label(softmax.dimension(0), softmax.dimension(1));
		real_label.setZero();
		assert(label.dimension(0) == inputs.dimension(0));
		for (int i = 0; i < label.dimension(0); i++)
		{
			real_label(i, label(i)) = 1;
		}

		TType losst = real_label*softmax.log();
		loss =-losst.mean();//交叉熵损失函数平均值，平均值反向求导的时候，需要记住梯度除以
		d_inputs = (softmax - real_label)*(1.f/float(inputs.size()));//由于loss计算的时候，我们一般是计算loss mean，所以反向求导的时候，需要除以(inputs.rows()*inputs.cols())* /

	}
	static void CSoftmaxLayer::test() {
		int batch_size = 4;
		int input_size = 3;
		int output_size = 2;


		float input_data[12] = {1, 2, 3, 6, 4, 5, 2, 8, 10, 12, 11, 9};
		Eigen::TensorMap<TType>inputs(input_data, batch_size, input_size);
		float label_data[4] = { 1, 0, 1, 1 };
		Eigen::TensorMap<Tensor1xf> label(label_data,batch_size);
		

		TType d_inputs;
		Tensor0xf loss;
		CSoftmaxLayer::softmax_loss_forward_backward(inputs, label, d_inputs, loss);
		std::cout << loss << std::endl;
		std::cout << d_inputs << std::endl;
	}
};

*/
