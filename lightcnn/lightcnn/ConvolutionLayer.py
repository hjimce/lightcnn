import  tensorflow as tf

batch_size = 3
input_channel = 2
input_height = 5
input_width = 5
output_channel = 2
kenel_height = 3
kenel_widht = 3

bottom=tf.constant([i*0.1 for i in range(batch_size*input_channel*input_height*input_width)],shape=(batch_size,input_height,input_width,input_channel),dtype=tf.float32)
weights=tf.constant([1./float(1+i) for i in range(output_channel*input_channel*kenel_height*kenel_widht)],shape=(kenel_height,kenel_widht,input_channel,output_channel),dtype=tf.float32)
bais=tf.constant([i*0.2 for i in range(output_channel)],shape=[output_channel],dtype=tf.float32)
label=tf.constant([1,0,1,1])

conv1=tf.nn.bias_add(tf.nn.conv2d(bottom,weights,strides=[1,1,1,1],padding='VALID'),bais)
#one_hot=tf.one_hot(label,2)
#predicts=tf.nn.softmax(relu_output)
#loss =-tf.reduce_mean(one_hot * tf.log(predicts))

#打印相关变量，梯度等，验证是否与c++结果相同
#d_output,d_inputs,d_weights,d_bais=tf.gradients(loss,[relu_output,inputs,weights,bais])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	conv1_data,bottom_data=sess.run([conv1,bottom])
	print (conv1_data)
	#print (bottom_data)

	'''print ('output',output_np)
	print('d_output', d_output_np)
	print ("d_inputs",d_inputs_np)
	print("d_weights", d_weights_np)
	print("d_bais", d_bais_np)'''
