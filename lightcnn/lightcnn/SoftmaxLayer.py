import  tensorflow as tf
inputs=tf.constant([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],dtype=tf.float32)
label=tf.constant([1,0,1,1])
one_hot=tf.one_hot(label,3)
predicts=tf.nn.softmax(inputs)
loss =-tf.reduce_mean(one_hot * tf.log(predicts))
gradient=tf.gradients(loss,inputs)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer());
	print (sess.run(loss))
