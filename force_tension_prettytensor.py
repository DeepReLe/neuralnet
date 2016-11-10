import csv
import prettytensor as pt
import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import math

#there are some weird issues with Tensorflow warnings popping up for things we're not doing
#TODO: figure out why that's happening
tf.logging.set_verbosity(tf.logging.ERROR)


# calls evalRow on each row, and then sorts by it
# trims the array, only keeping the lowest 10%
def preprocessRow(required_force):
  data = []
  for rowIndex in range(len(required_force)):
    data.append( (evalFitness(required_force[rowIndex]), required_force[rowIndex]))
  data.sort(key=lambda tup: tup[0])
  del data[(len(data)/10):]

  trimmed = []
  for i in data:
    trimmed.append(i[1])

  return trimmed

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



def trainData(f_out, learning_rate=0.01, iterations=100, batch_size=10, output_file='f2t_pt.txt'):
  global W_arr
  global bias

  #constants for the neural net
  plot_period = 1
  num_inputs = 3
  num_outputs = 1

  #reset the graph
  #this line allows trainData() to be called multiple times
  tf.reset_default_graph()

  x = tf.placeholder(tf.float32, shape=(None,num_inputs))
  y = tf.placeholder(tf.float32, shape=(None,num_outputs))

  init_normal = tf.random_normal_initializer(mean=0.0, stddev=0.01)

  loss = (pt.wrap(x)
        .flatten()
        .fully_connected(7, activation_fn=tf.nn.tanh, init=init_normal)
        .fully_connected(15, activation_fn=tf.nn.tanh, init=init_normal)
        .fully_connected(num_outputs, activation_fn=tf.nn.tanh, init=init_normal)
        .l2_regression(y))

  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = pt.apply_optimizer(optimizer, losses=[loss])

  validation_x = []
  validation_y = []
  init_op = tf.initialize_all_variables()

  #Fx Fy Fz range from [0,5]
  for F_x in range(6):
    for F_y in range(6):
      for F_z in range(6):
        inputForce = []
        inputForce.append(F_x)
        inputForce.append(F_y)
        inputForce.append(F_z)

       #calculate the offset, which is 0.8*number of rows for this given (Fx,Fy,Fz)
       #use this to only grab stuff in the last 20% of the dataset
       #learn on the first 80%
        offset = int(len(f_out[F_x][F_y][F_z])*0.8)
        for row in range(int(len(f_out[F_x][F_y][F_z])*0.2)):
          validation_x.append(inputForce)
          outputTension = []
          outputTension.append(f_out[F_x][F_y][F_z][row+offset][0])
          validation_y.append(outputTension)

  if os.path.isfile(output_file):
    os.remove(output_file)
  f = open(output_file, 'w')

  with tf.Session() as sess:
    sess.run(init_op)
    for i in xrange(iterations):

      #generate random batches
      muscle_activation_batch = []
      f_out_batch = []
      for j in xrange(batch_size):
        tempFx = random.randint(0,5);
        tempFy = random.randint(0,5);
        tempFz = random.randint(0,5);
        randRow = random.randint(1,len(f_out[tempFx][tempFy][tempFz])*0.8)

        muscleOut = []
        muscleOut.append(f_out[tempFx][tempFy][tempFz][randRow][0])
        muscle_activation_batch.append(muscleOut)

        forceVector = []
        forceVector.append(tempFx)
        forceVector.append(tempFy)
        forceVector.append(tempFz)
        f_out_batch.append(forceVector)

      mse = sess.run([train_op, loss],
                   {x: f_out_batch, y: muscle_activation_batch})

      #plot the data every plot_period
      if i%plot_period==0:
        validation_mse = sess.run([loss],
                    {x: validation_x,
                    y: validation_y})

        f.write(str(validation_mse[0]))
        f.write("\n")

  f.close()

def fullTraining(f_out=loadSubsampledData(), learning_rate=0.01, iterations=100, batch_size=10):
    #create a random file name
    #this allows fullTraining to run in parallel (i.e. two instances won't write to the same file)
    outputFile = str(random.random())
    trainData(f_out, learning_rate, iterations, batch_size, outputFile)

    f = open(outputFile, 'r')
    results = []
    for line in f:
      results.append(line)
    f.close()
    os.remove(outputFile)

    #return the result of the last run
    return float(results[len(results)-1])
