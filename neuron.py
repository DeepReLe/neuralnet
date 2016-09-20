import tensorflow as tf

graph = tf.get_default_graph()
operations = graph.get_operations()

x = tf.Variable(1.0)
w = tf.Variable(0.8)
y = x * w
y_ = tf.constant(0.0)
loss = (y-y_)**2

#op = graph.get_operations()[-1]

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss) # grad descent with learning rate of 0.025

for i in range(100): # run gradient descent 100 times
    sess.run(train_step)
    print sess.run(y)
print sess.run(y)
