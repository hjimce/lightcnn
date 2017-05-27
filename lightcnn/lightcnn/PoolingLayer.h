#pragma once
#include "config.h"
#include <vector>
enum PoolingMethod
{
	max,
	avg

};



class CPoolingLayer
{
public:

	CPoolingLayer(std::vector<int>pooling_shape,PaddingMethod padding_method,PoolingMethod pooling_method) {
		m_hksize = pooling_shape[0];
		m_wksize = pooling_shape[1];
		m_hstride = pooling_shape[2];
		m_wstride = pooling_shape[3];

		m_padding_method = padding_method;
		m_pooling_method = pooling_method;



		
	};
	~CPoolingLayer() {
	};
	//返回(bottom[0],bottom[1]/hstride*bottom[2]/wstride,hsize,wsize,bottom[3])
	void CPoolingLayer::extract_image_patches(const Tensor4xf &bottom, Tensor5xf &patches) {
		
		switch (m_padding_method)
		{
		case valid:
			patches = bottom.extract_image_patches(m_hksize, m_wksize, m_hstride, m_wstride, 1, 1, Eigen::PADDING_VALID);
			break;
		case same:
			patches = bottom.extract_image_patches(m_hksize, m_wksize, m_hstride, m_wstride, 1, 1, Eigen::PADDING_SAME);
			break;
		default:
			break;
		}
			
	}
	Eigen::DSizes<int, 4> CPoolingLayer::get_top_shape(const Tensor4xf&bottom) {
		Eigen::DSizes<int, 4>top_shape;
		top_shape[0] = bottom.dimension(0);
		switch (m_padding_method)
		{
		case valid:
			top_shape[1] = Eigen::divup(float(bottom.dimension(1) - m_hksize + 1), float(m_hstride));
			top_shape[2] = Eigen::divup(float(bottom.dimension(2) - m_wksize + 1), float(m_wstride));
			break;
		case same:
			top_shape[1] = Eigen::divup(float(bottom.dimension(1)), float(m_hstride));
			top_shape[2] = Eigen::divup(float(bottom.dimension(2)), float(m_wstride));

			break;
		default:
			break;
		}
		top_shape[3] = bottom.dimension(3);
		return top_shape;
	}


	void CPoolingLayer::forward(const Tensor4xf&bottom,Tensor4xf&top, const Eigen::ThreadPoolDevice &device) {
		Eigen::array<int, 2> reduction_dims{2,3};//第二维、第三维的大小等于（hksize、wksize）
		Eigen::DSizes<int, 4>post_reduce_dims=get_top_shape(bottom);
		Tensor5xf patches; //(bottom[0], bottom[1] / hstride*bottom[2] / wstride, hsize, wsize, bottom[3])
		extract_image_patches(bottom, patches);



		Tensor3xf pooling(post_reduce_dims[0],post_reduce_dims[1]*post_reduce_dims[2],post_reduce_dims[3]);
		switch (m_pooling_method)
		{
		case avg:
			pooling.device(device) = patches.mean(reduction_dims);//对reduction_dims内对应的维度索引进行统计，比如统计第3、2
			break;
		case max:
			pooling.device(device) = patches.maximum(reduction_dims);//最大池化
			break;
		default:
			break;
		}		
		top=pooling.reshape(post_reduce_dims);

	}
	void CPoolingLayer::backward(const Tensor4xf&bottom,const Tensor4xf&dtop, Tensor4xf&dbottom, const Eigen::ThreadPoolDevice &device) {
		dbottom.setZero();
		Tensor5xf patches; //(bottom[0], bottom[1] / hstride*bottom[2] / wstride, hsize, wsize, bottom[3])
		extract_image_patches(bottom, patches);
		//计算第2、3维的降维
		Eigen::array<Eigen::DenseIndex, 2> reduce_dims{2,3};
		auto index_tuples= patches.index_tuples();
		Eigen::Tensor<Eigen::Tuple<Eigen::DenseIndex, float>, 3, Eigen::RowMajor> reduced_by_dims;
		reduced_by_dims = index_tuples.reduce(reduce_dims, Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<Eigen::DenseIndex, float> >());
		
		//与forward阶段相反，需要reshape回去
		Tensor5xf dpatches(patches);
		dpatches.setZero();
		Eigen::DSizes<int, 4>post_reduce_dims = get_top_shape(bottom);
		Tensor3xf reshape = dtop.reshape(Eigen::array<int, 3>{post_reduce_dims[0], post_reduce_dims[1] * post_reduce_dims[2], post_reduce_dims[3]});
		//求dpatches
		for (int i=0;i<reduced_by_dims.size();i++)//最大池化反向求导
		{
			const Eigen::Tuple<Eigen::DenseIndex, float>& v = reduced_by_dims(i);
			dpatches.data()[v.first] = reshape.data()[i];
		}
		//求dbottom
		//auto dp_reshape = dpatches.reshape(dpatches.dimension(0),m_hksize,m_wksize);
		for (int i=0;i<dpatches.dimension(0);i++)
		{
			for (int j=0;j<dpatches.dimension(1);j++)
			{
				int height = j / post_reduce_dims[1];
				int width = j% post_reduce_dims[1];
				for (int k=0;k<dpatches.dimension(4);k++)
				{
					
					//把每个dpathches的pathces数值加到dbottom中
					for (int hp=0;hp<dpatches.dimension(2);hp++)
					{
						for (int wp = 0; wp<dpatches.dimension(3); wp++)
						{
							auto &element_dbottom = dbottom(i, m_hstride*height+ hp, m_wstride*width + wp, k);
							element_dbottom += dpatches(i, j, hp, wp, k);
						}
					}

					//std::cout << reduced_by_dims(i,j,k).<<":"<<reduced_by_dims(i,j,k).second<<std::endl;
				}
			}
			
		}


		//std::cout << "reduced_by_dims" << reduced_by_dims << std::endl;


	}
private:
	int m_hksize;
	int m_wksize;
	int m_hstride;
	int m_wstride;
	
	PaddingMethod m_padding_method;
	PoolingMethod m_pooling_method;

};


class CPoolingLayer_test
{
public:
	static void CPoolingLayer_test::test() {


		Eigen::ThreadPool *tp = new Eigen::ThreadPool(8);
		Eigen::ThreadPoolDevice device(tp, 8);

		int batch_size = 1;
		int input_channel = 1;
		int input_height = 5;
		int input_width = 5;
		int kenel_height = 2;
		int kenel_widht = 2;
		int kstride = 3;
		float *input_data = new float[batch_size*input_channel*input_height*input_width];
		for (int i = 0; i < batch_size*input_channel*input_height*input_width; i++)
		{
			input_data[i] = 0.1*i;
		}
		Eigen::TensorMap<Tensor4xf>bottom(input_data, batch_size, input_height, input_width, input_channel);


		Tensor1xf label_1d(batch_size);
		label_1d.setZero();
		label_1d.setValues({ 1, 0, 2});
		Tensor2xf one_hot;
		


		//第一层：pooling层
		CPoolingLayer layer({kenel_height,kenel_widht,kstride,kstride},PaddingMethod::same,PoolingMethod::max);
		//Tensor4xf bottom(batch_size,input_height,input_width, input_channel);

		Tensor4xf top;
		layer.forward(bottom, top,device);
/*
		Tensor2xf top_flatten;
		CBaseFunction::flatten(top, top_flatten);


		//第二层：sotfmax网络层
		CBaseFunction::onehot(label_1d, top_flatten.dimension(1), one_hot);
		Tensor2xf dtop_flatten(top_flatten);
		float loss = CBaseFunction::softmax_with_loss(top_flatten, one_hot, dtop_flatten, device);
		Tensor4xf dtop;
		CBaseFunction::reshape_like(dtop_flatten, top, dtop);



		Tensor4xf dbottom(bottom);

		layer.backward(bottom, dtop,dbottom,device);*/



		std::cout << "***************forward************" << std::endl;
		CBaseFunction::print_shape(one_hot);
		CBaseFunction::print_shape(top);
		CBaseFunction::print_element(top);
		std::cout << "bottom" << bottom<< std::endl;
		//std::cout << "top" << top << std::endl;
		//std::cout << "dbottom" << dbottom << std::endl;
		//std::cout << "loss" << loss << std::endl;
		
		//std::cout << "dbottom" << dbottom << std::endl;
		//std::cout << "dtop" << top << std::endl;





	}

};