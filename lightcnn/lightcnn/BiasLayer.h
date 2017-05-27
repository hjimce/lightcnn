#pragma once
#include "config.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
template <class TType>
class CBiasLayer
{
public:
	CBiasLayer(Eigen::ThreadPoolDevice *device) {
		m_device = device;
		m_bias = NULL;
		m_top = NULL;

		m_dbias = NULL;
		m_dbottom = NULL;

	};
	~CBiasLayer() {
		delete m_bias;
		delete m_top;
		delete m_dbias;
		delete m_dbottom;

	};
	void CBiasLayer::forward_initialise(TType *bottom) {
		int bias_size = bottom->dimension(bottom->NumDimensions - 1);
		m_bottom = bottom;
		m_bias = new Tensor1xf(bias_size);
		m_top = new TType(*bottom);


		m_bias->setRandom();
		m_top->setZero();


	}
	void CBiasLayer::backward_initialise(TType *dtop){
		m_dtop = dtop;
		int bias_size = dtop->dimension(dtop->NumDimensions - 1);
		m_dbias = new Tensor1xf(bias_size);
		m_dbottom = new TType(*dtop);
		m_dbias->setZero();
		m_dbottom->setZero();

	}


	void CBiasLayer::backward() {
		switch (m_top->NumDimensions)
		{
		case 2:
			m_dbias->device(*m_device) = m_dtop->sum(Eigen::array<int, 1> { 0 });//eigen::tensor 类好像不支持动态reduce opt，所以这边暂时采用switch
			break;
		case 3:
			m_dbias->device(*m_device) = m_dtop->sum(Eigen::array<int, 2> { 0,1 });
			break;
		case 4:
			m_dbias->device(*m_device) = m_dtop->sum(Eigen::array<int, 3> { 0, 1,2 });
			break;
		default:
			break;
		}
		//
		m_dbottom->device(*m_device) = *m_dtop;
	}

public:
	Eigen::ThreadPoolDevice *m_device;
	TType *m_bottom;
	Tensor1xf *m_bias;
	TType *m_top;

	TType*m_dtop;
	Tensor1xf*m_dbias;
	TType*m_dbottom;





};
class CBiasLayer_test
{
public:
	static void CBiasLayer_test::test() {

/*
		Eigen::ThreadPool *tp=new Eigen::ThreadPool(8);
		Eigen::ThreadPoolDevice *temp=new Eigen::ThreadPoolDevice(tp, 8);



		CBiasLayer<Tensor3xf> biaslayer(temp);
		Tensor3xf *bottom=new Tensor3xf(2, 1,3);
		bottom->setConstant(1);

		std::cout << "***************forward************" << std::endl;
		biaslayer.forward_initialise(bottom);
		biaslayer.forward();

		std::cout << *biaslayer.m_bias << std::endl;
		std::cout << *biaslayer.m_top << std::endl;


		Tensor3xf *dtop=new Tensor3xf(2,1,3);
		dtop->setConstant(2);
		biaslayer.backward_initialise(dtop);
		biaslayer.backward();



		std::cout << "***************backward************" << std::endl;
		std::cout << *biaslayer.m_dbias << std::endl;
		std::cout << *biaslayer.m_dbottom << std::endl;


		delete tp;
		delete temp;
		delete bottom;
		delete dtop;

*/






	}

};

