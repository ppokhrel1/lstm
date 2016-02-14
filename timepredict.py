from pybrain.datasets import SequentialDataSet
from itertools import cycle

ds = SequentialDataSet(1, 1)
with open('training.txt') as file:
    data = [1.0, 0.0]
    for line in file:
        for x in list(line):
            data.append(ord(x))
#data  = [1] * 3 + [2] * 3
for sample, next_sample in zip(data, cycle(data[:1])):
    ds.addSample(sample, next_sample)

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer

net = buildNetwork(1, 100, 1, hiddenclass=LSTMLayer, outputbias=False, recurrent=True)

from pybrain.supervised import RPropMinusTrainer
from sys import stdout

trainer = RPropMinusTrainer(net, dataset=ds)
train_errors = [] # save errors for plotting later
EPOCHS_PER_CYCLE = 5
CYCLES = 100
EPOCHS = EPOCHS_PER_CYCLE * CYCLES
for i in xrange(CYCLES):
    trainer.trainEpochs(EPOCHS_PER_CYCLE)
    train_errors.append(trainer.testOnData())
    epoch = (i+1) * EPOCHS_PER_CYCLE
    print "\r epoch {}/{}".format(epoch, EPOCHS)
    stdout.flush()

    print "final error =", train_errors[-1]

import matplotlib.pyplot as plt

plt.plot(range(0, EPOCHS, EPOCHS_PER_CYCLE), train_errors)
plt.xlabel('epoch')
plt.ylabel('error')
plt.show()

for sample, target in ds.getSequenceIterator(0):
    print "               sample = %4.1f" % sample
    print "predicted next sample = %4.1f" % net.activate(sample)
    print "   actual next sample = %4.1f" % target
myinput = input("Enter a letter")
data = net.activate(ord(myinput))
print chr(data)
while True:
    data = net.activate(data)
    print chr(data)

import cPickle as pickle
with open('nn.pkl', wb) as f:
    pickle.dump(net, f, pickle.HIGHEST_PROTOCOL)
