import  tensorflow as tf
inputs=tf.constant([1,2,3,6,4,5,2,8,10,12,11,9],shape=(4,3),dtype=tf.float32)
weights=tf.constant([0.55, 0.88, 0.75, 1.1, 0.11, 0.002],shape=(3,2),dtype=tf.float32)
bais=tf.constant([3, 2],dtype=tf.float32)
label=tf.constant([1,0,1,1])

output=tf.matmul(inputs,weights)+bais
one_hot=tf.one_hot(label,2)
predicts=tf.nn.softmax(output)
loss =-tf.reduce_mean(one_hot * tf.log(predicts))


d_output,d_inputs,d_weights,d_bais=tf.gradients(loss,[output,inputs,weights,bais])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	loss_np,output_np,d_output_np, d_inputs_np, d_weights_np, d_bais_np=sess.run([loss,output,d_output,
	                                                                      d_inputs,d_weights,d_bais])
	print (loss_np)
	print ('output',output_np)
	print('d_output', d_output_np)
	print ("d_inputs",d_inputs_np)
	print("d_weights", d_weights_np)
	print("d_bais", d_bais_np)
