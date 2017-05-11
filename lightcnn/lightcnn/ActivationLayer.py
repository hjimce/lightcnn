import  tensorflow as tf
inputs=tf.constant([[1,-2,3],[4,-5,6],[7,-8,-9],[10,11,-12]],shape=(4,3),dtype=tf.float32)
weights=tf.constant([0.55, -0.88, 0.75, -1.1, -0.11, 0.002],shape=(3,2),dtype=tf.float32)
bais=tf.constant([3, -2],dtype=tf.float32)
label=tf.constant([1,0,1,1])

relu_output=tf.nn.relu(tf.matmul(inputs,weights)+bais)
one_hot=tf.one_hot(label,2)
predicts=tf.nn.softmax(relu_output)
loss =-tf.reduce_mean(one_hot * tf.log(predicts))


d_output,d_inputs,d_weights,d_bais=tf.gradients(loss,[relu_output,inputs,weights,bais])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	loss_np,output_np,d_output_np, d_inputs_np, d_weights_np, d_bais_np=sess.run([loss,relu_output,d_output,
	                                                                      d_inputs,d_weights,d_bais])
	print (loss_np)
	print ('output',output_np)
	print('d_output', d_output_np)
	print ("d_inputs",d_inputs_np)
	print("d_weights", d_weights_np)
	print("d_bais", d_bais_np)
