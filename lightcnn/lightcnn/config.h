#pragma once
#include <unsupported/Eigen/CXX11/ThreadPool>
#include <unsupported/Eigen/CXX11/Tensor>

typedef Eigen::Tensor<float,0, Eigen::ColMajor> Tensor0xf;
typedef Eigen::Tensor<float, 1, Eigen::ColMajor> Tensor1xf;
typedef Eigen::Tensor<float, 2, Eigen::ColMajor> Tensor2xf;
typedef Eigen::Tensor<float, 3, Eigen::ColMajor> Tensor3xf;
typedef Eigen::Tensor<float, 4, Eigen::ColMajor> Tensor4xf;
typedef Eigen::Tensor<float, 5, Eigen::ColMajor> Tensor5xf;



typedef Eigen::Tensor<float, 1, Eigen::RowMajor> Tensor1rf;
typedef Eigen::Tensor<float, 2, Eigen::RowMajor> Tensor2rf;
typedef Eigen::Tensor<float, 3, Eigen::RowMajor> Tensor3rf;
typedef Eigen::Tensor<float, 4, Eigen::RowMajor> Tensor4rf;
typedef Eigen::Tensor<float, 5, Eigen::RowMajor> Tensor5rf;

#define  EIGEN_USE_THREADS 8



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




