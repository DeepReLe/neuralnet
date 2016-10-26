import randomSearch as rs


name = "exp1"
lo = 0.001
hi = 0.1
mean = 0
std = 0.1

experiment = rs.Experiment(name, {
"learning_rate" : rs.uniform(lo, hi),
"epochs" : [50, 60, 75, 100],
"batch_size" : rs.normal(mean, std)
# additional variables
})

#experiment.seed(1234)


for i in xrange(1000):
    loss, convergence = train_net(experiment.learning_rate, experiment.epochs, experiment.batch_size)
    experiment.saveResult(loss, additional_data = {
    "convergence" : convergence
    })

def train_net(lr, epochs, size):
    return loss, convergence

lr, epochs, size = experiment.optimal_result()

# save to disk as json
# web page reads json info
