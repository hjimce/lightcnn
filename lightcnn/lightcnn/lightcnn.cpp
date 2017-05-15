// lightcnn.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#include "SoftmaxLayer.h"
#include "FullyconnecteLayer.h"
#include "ActivationLayer.h"
#include "ConvolutionLayer.h"


int main()
{
	//CSoftmaxLayer::test();
	//CFullyconnecteLayer::test();
	CActivationLayer::test();
	//CConvolutionLayer::test();
/*
	Tensor2xf t(4, 3);
	t.setRandom();

	Eigen::Tensor<float, 0,Eigen::RowMajor> frob_norm_tens = t.square().sum();
	const float frob_norm = frob_norm_tens.coeff();
	std::cout << frob_norm << std::endl;*/

	return 0;




    return 0;
}

