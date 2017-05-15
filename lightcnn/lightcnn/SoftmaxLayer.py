import  tensorflow as tf

batch_size = 4
input_size = 3
output_size = 2
inputs=tf.constant([1,2,3,6,4,5,2,8,10,12,11,9],shape=[batch_size,input_size],dtype=tf.float32)
label=tf.constant([1,0,1,1])
one_hot=tf.one_hot(label,3)
predicts=tf.nn.softmax(inputs)
loss =-tf.reduce_mean(one_hot * tf.log(predicts))
gradient=tf.gradients(loss,inputs)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer());
	print (sess.run(loss))
	print (sess.run(gradient))
