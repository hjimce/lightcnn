#pragma once
#include "config.h"
class CBaseFunction
{
public:
	static void CBaseFunction::softmax(const Tensor2xf &inputs, Tensor2xf &softmax,const Eigen::ThreadPoolDevice &device) {
		softmax = inputs;
		softmax.device(device)= inputs.exp();
		auto sorfmax_rowsum= softmax.sum(Eigen::array<int, 1>{ {1}}).eval().reshape(Eigen::array<int, 2>{ {inputs.dimension(0), 1}});//第一维降维,这边不能用auto，会出现计算数值错误
		auto sorfmax_rowsum_broad = sorfmax_rowsum.broadcast(Eigen::array<int, 2> { {1, inputs.dimension(1)}});

		softmax.device(device) = softmax / sorfmax_rowsum_broad;//行归一化
	}
	static float CBaseFunction::softmax_with_loss(const Tensor2xf &bottom, const Tensor2xf&real_label, Tensor2xf &dbottom,
		const Eigen::ThreadPoolDevice &device
	) {
		Tensor2xf top;
		CBaseFunction::softmax(bottom, top,device);
		Tensor0xf loss;
		loss.device(device) = -(real_label*(top.log()).eval()).mean();//交叉熵损失函数平均值，平均值反向求导的时候，需要记住梯度除以

		dbottom.device(device) = (top - real_label)*(1.f / float(top.size()));//由于loss计算的时候，我们一般是计算loss mean，所以反向求导的时候，需要除以(inputs.rows()*inputs.cols())* /

		return loss(0);
	}




	template <class TType>
	static void CBaseFunction::add_bias(const TType& bottom, const Tensor1xf &bias, TType& top,
		const Eigen::ThreadPoolDevice &device) {
		const int bias_size = bias.dimension(0);
		const int rest_size = bottom.size() / bias_size;
		Eigen::DSizes<int, 1> one_d(bottom.size());
		Eigen::DSizes<int, 1> bcast(rest_size);

		top.reshape(one_d).device(device) = bottom.reshape(one_d) + bias.broadcast(bcast).reshape(one_d);

	}

	static void CBaseFunction::flatten(const Tensor4xf& bottom, Tensor2xf &top) {
		const int first_dim = bottom.dimension(0);
		const int second_dim= bottom.size()/bottom.dimension(0);
		
		top = bottom.reshape(Eigen::array<int, 2>{ first_dim,second_dim });
	}
	//template <class TType1, class TType2>
	static void CBaseFunction::reshape_like(const Tensor2xf &src,const Tensor4xf&dst,Tensor4xf &output) {

		/*std::string t;
		for (int i = 0; i < tensor.NumDimensions; i++)
		{
			t += std::to_string(tensor.dimension(i)) + " ";
		}
		std::cout << "tensor shape:" << t << std::endl;*/
		output = src.reshape(Eigen::array<int, dst.NumDimensions>{ dst.dimension(0), dst.dimension(1), dst.dimension(2), dst.dimension(3) });

	}







	static void CBaseFunction::onehot(const Tensor1xf &bottom,const int &depth ,Tensor2xf &top){
		top.resize(bottom.size(), depth);
		top.setZero();
		for (int i = 0; i < bottom.dimension(0); i++)
		{
			if (bottom(i)>=depth)
			{
				std::cout << "error in line"<<i<<":label out of range " << std::endl;
			}
			top(i, bottom(i)) = 1;
		}
	}
	template <class TType>
	static void CBaseFunction::print_shape(const TType& tensor) {
		std::string t;
		for (int i=0;i<tensor.NumDimensions;i++)
		{
			t += std::to_string(tensor.dimension(i))+" ";
		}
		std::cout <<"tensor shape:" <<t << std::endl;

	}
	template <class TType>
	static void CBaseFunction::print_element(const TType& tensor) {
		//std::cout << "tensor element:" << tensor << std::endl;
		switch (tensor.NumDimensions)
		{
		case 2:
/*
			for (int i = 0; i < tensor.dimension(0); i++)
			{
				for (int j = 0; j < tensor.dimension(1); j++)
				{
					std::cout << "tensor element:" << tensor(i, j) << std::endl;
				}

			}*/
			break;

		case 4:

			for (int i = 0; i < tensor.dimension(0); i++)
			{
				for (int j = 0; j < tensor.dimension(1); j++)
				{
					for (int k = 0; k < tensor.dimension(2); k++)
					{
						for (int h = 0; h < tensor.dimension(3); h++)
						{
							std::cout << "tensor element:" << tensor(i, j, k, h) << std::endl;
						}
					}
				}
			}
			break;

		default:
			break;
		}




	}
};

enum PaddingMethod
{
	valid,
	same

};