__author__ = 'Michael'
# Class definition for Neural Nets
# There are 3 layers
# -top- neural net (a Collection of layers)
# -mid- layer (a collection of neurons)
# -low- neuron (a collection of links)
import random

class Neuron(object):
    def __init__(self):
        self.affectors = {}                      # contains a link to a neuron and an associated weight.
        self.value = -1                          # default out-of bounds value.
    def addAffector(self, n, weight):
        self.affectors[n] = weight

    def adjustWeightByFactor(self, n, factor):      # adjusts the source neurons wieght by that factor. (0.3 means 30%) used for making the network learn.
        self.affectors[n] = self.affectors[n] * factor

    def setWeight(self, n, weight):
        self.affectors[n] = weight

    def setValue(self, value):
        self.value = value

    def calcValue(self):
        self.value = 0                            # this is the accumulator
        for i in self.affectors.keys():
            self.value += i.value * self.affectors[i]      # add the wieights together.

    def randomize(self):                #Randomizes current value and weights associated with each source neuron.
        self.value = random.random()                            # randomize value.. if wieghts are to be used without first calling calcValue
        for i in self.affectors.keys():                         # for every link...
            self.affectors[i] = random.random()                 # assign a value between 0 and 1

class Layer(object):
    def __init__(self):
        self.neuronList = []

    def addNeuron(self, neuron):
        self.neuronList.add(neuron)

    def calcLayer(self):                # Make every Neuron in this layer calculate its new value.
        for n in self.neuronList:
            n.calcValue()

class Net(object):
    add = 1





