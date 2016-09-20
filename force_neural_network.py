import csv
import tensorflow as tf
import numpy as np
import random

data = []

with open('data/matrix.csv') as csvfile:
    rowReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in rowReader:
        data.append(', '.join(row).split(','))

data = data[1:]

#muscle_activation = np.array([])
#f_out = np.array([])
muscle_activation = []
f_out = []
for i in data:
    #a = np.array([i[2], i[4], i[6], i[8], i[10], i[12], i[14]])
    #np.append(muscle_activation, a)
    #np.append(f_out, i[18])
    a = []
    for j in range(15):
        if j%2==0 and j!=0:
            a.append(i[j])
    muscle_activation.append(a)
    #muscle_activation.append([tf.placeholder(i[2]), i[4], i[6], i[8], i[10], i[12], i[14]])
    #f_out.append([i[18], i[16], i[21], i[17], i[20], i[19]]) # F_x, F_y, F_z, M_x, M_y, M_z
    f_out.append([i[18]])

x = tf.placeholder(tf.float32, shape=(None, 7))
W = tf.Variable(tf.zeros([7,1]))
b = tf.Variable(tf.zeros([1]))

y = tf.matmul(x,W) + b

y_ = tf.placeholder(tf.float32,  shape=(None,1))

cost = tf.reduce_mean(tf.square(y-y_))


train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(cost)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000): #iterations
    muscle_activation_batch = []
    f_out_batch = []
    for k in range(100):
        temprand = random.randint(1,10000)
        muscle_activation_batch.append(muscle_activation[temprand])
        f_out_batch.append(f_out[temprand])

    feed = {x: muscle_activation_batch, y_: f_out_batch}
    sess.run(train_step, feed_dict=feed)

    print ("After %d iteration:" %i)
    print "W:"
    print sess.run(W)
    print "b"
    print sess.run(b)


W_arr = sess.run(W)
bias = sess.run(b)[0]

mean_square_sum_error = 0
average_percentage_error = 0


#validation set

print "ERROR: "
for i in range(10001,14000):
    expected_value = float(data[i][18])
    predicted_value = 0
    predicted_muscle_activations = []
    for j in range(15):
        if j%2==0 and j!=0:
            predicted_muscle_activations.append(float(data[i][j]))

    for j,k in enumerate(W_arr):
        predicted_value += predicted_muscle_activations[j]*k

    predicted_value+=bias
    average_percentage_error += abs(predicted_value-expected_value)/expected_value
    mean_square_sum_error += (predicted_value-expected_value)**2
    print "predicted: " , predicted_value , " actual: " , expected_value

mean_square_sum_error/=4000
average_percentage_error/=4000

print "Mean Square Sum Error: "
print mean_square_sum_error

print "Average Error: "
print average_percentage_error
