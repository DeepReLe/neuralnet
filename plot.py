import matplotlib.pyplot as plt

data = []

f = open('pt_14k.txt', 'r')

for line in f:
    data.append(float(line))


plt.title('Two Layered Neural Net')
plt.plot(data)
plt.xlabel('Iterations')
plt.ylabel('L2 Regression')
plt.show()
