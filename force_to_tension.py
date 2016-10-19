#linear regression on larger dataset
# trying to train F_x, F_y, and F_z at the same time -> ini
import os
import csv
import tensorflow as tf
import numpy as np
import random
import time

iterations = 100
batch_size = 100
num_inputs = 3
num_outputs = 7
step_size = 0.001



# calls evalRow on each row, and then sorts by it
# trims the array, only keeping the lowest 10%
def preprocessRow(required_force):
    data = []
    for rowIndex in range(len(required_force)):
        data.append( (evalFitness(required_force[rowIndex]), required_force[rowIndex]))
    data.sort(key=lambda tup: tup[0])
    del data[(len(data)/10):]

    #this is a hacky way to convert from a tuple to a list
    trimmed = []
    for i in data:
        trimmed.append(i[1])

    return trimmed

# finds the tension of a given row
def evalFitness(row):
    value = 0
    for tension in row:
       value += float(tension)*float(tension)
    return value

# forces_required[a][b][c] returns a list of muscle tension variations that produce
# f_x = a, f_y = b, and f_z = c
def loadSubsampledData():
    forces_required = [[[0 for a in range(6)] for b in range(6)] for c in range(6)]
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
            forces_required[int(f_x)][int(f_y)][int(f_z)] = preprocessRow(data_solo)
    return forces_required

forces_required = loadSubsampledData();
print len(forces_required[0][0][0])

# x = tf.placeholder(tf.float32, shape=(None, num_inputs))
# W = tf.Variable(tf.zeros([num_inputs,num_outputs]))
# b = tf.Variable(tf.zeros([num_outputs]))

# y = tf.matmul(x,W) + b

# y_ = tf.placeholder(tf.float32, shape=(None,num_outputs))

# cost = tf.reduce_mean(tf.square(y-y_))


# train_step = tf.train.GradientDescentOptimizer(step_size).minimize(cost)
# init = tf.initialize_all_variables()

# sess = tf.Session()
# sess.run(init)

# for i in range(iterations): #iterations
#     muscle_activation_batch = []
#     f_out_batch = []

#     for j in range(batch_size):

#         # select a random force vector
#         F_x = random.randint(0,5);
#         F_y = random.randint(0,5);
#         F_z = random.randint(0,5);
#         f_out_batch.append([F_x, F_y, F_z])

#         #select a random row of valid activations for that force vector
#         randRow = random.randint(0, len(forces_required[F_x][F_y][F_z])*0.8-1)
#         muscle_activation_batch.append(forces_required[F_x][F_y][F_z][randRow])

#     feed = {x: f_out_batch, y_: muscle_activation_batch}
#     sess.run(train_step, feed_dict=feed)

#     print ("After %d iteration:" %i)
#     print "W:"
#     print sess.run(W)
#     print "b"
#     print sess.run(b)

# W_arr = sess.run(W)
# W_arr_fx = []
# for i in W_arr:
#     W_arr_fx.append(i[0])


# bias = sess.run(b)[0]
# print "ERROR: "

# mean_square_sum_error = 0
# average_percentage_error = 0

# offset = int(len(forces_required[F_x][F_y][F_z])*0.8)
# count = 0
# for F_x in range(6):
#     for F_y in range(6):
#         for F_z in range(6):
#             for row in range(int(len(forces_required[F_x][F_y][F_z])*0.2)):
#                 count += 1
#                 expected_value = forces_required[F_x][F_y][F_z][row+offset][0]
#                 predicted_value = 0
#                 inputForce = []
#                 for j in range(3):
#                     inputForce.append(F_x)
#                     inputForce.append(F_y)
#                     inputForce.append(F_z)

#                 for j,k in enumerate(W_arr_fx):
#                     predicted_value += float(inputForce[j]) * k
#                 predicted_value+=bias
#                 if expected_value == 0:
#                     average_percentage_error += abs(float(predicted_value))
#                     mean_square_sum_error += (float(predicted_value)-float(expected_value))**2
#                 else:
#                     average_percentage_error += abs(float(predicted_value)-float(expected_value))/float(expected_value)
#                     mean_square_sum_error += (float(predicted_value)-float(expected_value))**2
#                 # print "predicted: " , predicted_value , " actual: " , expected_value

# mean_square_sum_error/=count*36
# average_percentage_error/=count*36

# print "Mean Square Sum Error: "
# print mean_square_sum_error

# print "Average Error: "
# print average_percentage_error
