import  tensorflow as tf

batch_size = 1
input_channel = 1
input_height = 5
input_width = 5

kenel_height = 2
kenel_widht = 2
kstride = 3
bottom=tf.constant([i*0.1 for i in range(batch_size*input_channel*input_height*input_width)],shape=(batch_size,input_height,input_width,input_channel),dtype=tf.float32)



pool1=tf.nn.max_pool(bottom,[1,kenel_height,kenel_widht,1],strides=[1,kstride,kstride,1],padding='SAME')
pool_flatten=tf.reshape(pool1,[batch_size,-1])

label=tf.constant([1,0,3])
one_hot=tf.one_hot(label,pool_flatten.get_shape().as_list()[1])


predicts=tf.nn.softmax(pool_flatten)
loss =-tf.reduce_mean(one_hot * tf.log(predicts))

#打印相关变量，梯度等，验证是否与c++结果相同
dbottom=tf.gradients(loss,[bottom])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	bottom,pool1_data,dbottom_data,loss=sess.run([bottom,pool1,dbottom,loss])
	#print (bottom)
	print (pool1_data)
	#print (loss)

	#print ('dbottom_data',dbottom_data)

