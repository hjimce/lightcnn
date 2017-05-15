#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

typedef Eigen::Tensor<float,0, Eigen::RowMajor> Tensor0xf;
typedef Eigen::Tensor<float, 1, Eigen::RowMajor> Tensor1xf;
typedef Eigen::Tensor<float, 2, Eigen::RowMajor> Tensor2xf;
typedef Eigen::Tensor<float, 3, Eigen::RowMajor> Tensor3xf;
typedef Eigen::Tensor<float, 4, Eigen::RowMajor> Tensor4xf;


/*
template <typename Device, typename T, int Dims>
void operator()(const Device& d, T input,T bias,
		typename TTypes<T, Dims>::Tensor output)
	{
		const int bias_size = bias.dimension(0);
		const int rest_size = input.size() / bias_size;
		Eigen::DSizes<int, 1> one_d(input.size());
		Eigen::DSizes<int, 1> bcast(rest_size);
		To32Bit(output).reshape(one_d).device(d) =
		To32Bit(input).reshape(one_d) +
		To32Bit(bias).broadcast(bcast).reshape(one_d);

	}*/




