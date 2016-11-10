#linear regression on larger dataset
# trying to train F_x, F_y, and F_z at the same time -> ini
import os
import csv
import tensorflow as tf
import numpy as np
import random

iterations = 10000
batch_size = 100
num_inputs = 7
num_outputs = 3
step_size = 1

forces_required = [[[0 for a in range(6)] for b in range(6)] for c in range(6)]

# forces_required[a][b][c] returns a list of muscle tension variations that produce
# f_x = a, f_y = b, and f_z = c

i = len("activations_")
for filename in os.listdir('data/subsampled'):
    if filename[0:1] == 'a':
        f_x = float(filename[i:i+3])
        f_y = float(filename[i+4:i+7])
        f_z = float(filename[i+8:i+11])
        forces = [f_x, f_y, f_z]
        data_solo = []
        with open('data/subsampled/' + filename) as csvfile:
            rowReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in rowReader:
                temp = ', '.join(row).split(',')
                data_solo.append(temp)
        forces_required[int(f_x)][int(f_y)][int(f_z)] = data_solo



x = tf.placeholder(tf.float32, shape=(None, num_inputs))
W = tf.Variable(tf.zeros([num_inputs,num_outputs]))
b = tf.Variable(tf.zeros([num_outputs]))

y = tf.matmul(x,W) + b

y_ = tf.placeholder(tf.float32, shape=(None,num_outputs))

cost = tf.reduce_mean(tf.square(y-y_))


train_step = tf.train.GradientDescentOptimizer(step_size).minimize(cost)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(iterations): #iterations
    muscle_activation_batch = []
    f_out_batch = []

    F_x = random.randint(0,5);
    F_y = random.randint(0,5);
    F_z = random.randint(0,5);

    for j in range(batch_size):
        randRow = random.randint(0, len(forces_required[F_x][F_y][F_z])*0.8-1)
        muscle_activation_batch.append(forces_required[F_x][F_y][F_z][randRow])


    f_out_batch.append([F_x, F_y, F_z])

    feed = {x: muscle_activation_batch, y_: f_out_batch}
    sess.run(train_step, feed_dict=feed)

    print ("After %d iteration:" %i)
    print "W:"
    print sess.run(W)
    print "b"
    print sess.run(b)


W_arr = sess.run(W)
bias = sess.run(b)[0]

##################################################################
'''
Error check on 14k lines of data
'''


muscle_activation = []
f_out = []
data = []

with open('data/matrix.csv') as csvfile:
    rowReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in rowReader:
        data.append(', '.join(row).split(','))

data = data[1:]

#muscle_activation = np.array([])
#f_out = np.array([])

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


W_arr = sess.run(W)
W_arr_fx = []
for i in W_arr:
    W_arr_fx.append(i[0])


bias = sess.run(b)[0]
print "ERROR: "

mean_square_sum_error = 0
average_percentage_error = 0

for i in range(800, 1000):
    for x in range(0,5):
        for y in range(0,5):
            for z in range(0,5):
                expected_value = x
                predicted_value = 0
                predicted_muscle_activations = []
                for j in range(7):
                    predicted_muscle_activations.append(forces_required[x][y][z][i][j])
                for j,k in enumerate(W_arr_fx):
                    predicted_value += float(predicted_muscle_activations[j]) * k
                predicted_value+=bias
                if expected_value == 0:
                    average_percentage_error += abs(predicted_value)
                    mean_square_sum_error += (predicted_value-expected_value)**2
                else:
                    average_percentage_error += abs(predicted_value-expected_value)/expected_value
                    mean_square_sum_error += (predicted_value-expected_value)**2
                print "predicted: " , predicted_value , " actual: " , expected_value


mean_square_sum_error/=200*36
average_percentage_error/=200*36

print "Mean Square Sum Error: "
print mean_square_sum_error

print "Average Error: "
print average_percentage_error
