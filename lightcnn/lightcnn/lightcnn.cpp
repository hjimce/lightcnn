// lightcnn.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include "BiasLayer.h"

#include "SoftmaxLayer.h"
#include "FullyconnecteLayer.h"
#include "ActivationLayer.h"
#include "ConvolutionLayer.h"
#include "PoolingLayer.h"


int main()
{
	//CSoftmaxLayer_test::test();
	//CFullyconnecteLayer_test::test();
	//CActivationLayer_test::test();
	CPoolingLayer_test::test();
	//CConvolutionLayer_test::test();
	//CBiasLayer_test::test();

/*
	Tensor2xf t(4, 3);
	t.setRandom();

	Tensor2xf flipt = t.inverse();



	std::cout << t << std::endl; 
	std::cout << flipt << std::endl;*/






    return 0;
}

