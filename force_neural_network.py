import csv
import tensorflow as tf


data = []

with open('data/matrix.csv') as csvfile:
    rowReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in rowReader:
        data.append(', '.join(row).split(','))

data = data[1:]

muscle_activation = []
f_out = []
for i in data:
    muscle_activation.append([i[2], i[4], i[6], i[8], i[10], i[12], i[14]])
    f_out.append([i[18], i[16], i[21], i[17], i[20], i[19]]) # F_x, F_y, F_z, M_x, M_y, M_z
