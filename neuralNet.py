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
        self.threshold = 1

    def addAffector(self, n, weight = None):
        if weight == None:
            weight = random.random()
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
        self.value = self.value / self.threshold
        #print (self.value, self.threshold) # for debug
        if self.value > 0.9:
            self.value = 0.9
        if self.value < 0.1:
            self.value = 0.1

    def randomize(self):                #Randomizes current value and weights associated with each source neuron.
        self.value = random.random()                            # randomize value.. if wieghts are to be used without first calling calcValue
        self.threshold = random.randint(10, 884)                 # not sure what to start these on..
        for i in self.affectors.keys():                         # for every link...
            self.affectors[i] = random.random()                 # assign a value between 0 and 1

class Layer(object):
    def __init__(self, layerBelow = None, layerSize = None):
        if layerBelow == None :
            self.neuronList = []
            self.count = 0
        else:
            self.neuronList = []
            self.count = layerSize
            self.initNeurons(layerSize)
            self.connectFully(layerBelow)

    def addNeuron(self, neuron):
        self.neuronList.append(neuron)
        self.count += 1

    def initNeurons(self, layerSize):
        for i in range(0,layerSize):
            self.addNeuron(Neuron())

    def connectFully(self, layerBelow):
        for i in self.neuronList:
            for lbNeuron in layerBelow.neuronList:
                i.addAffector(lbNeuron)         # add with random weight.

    def setValues(self, valueList):
        if len(valueList) != len(self.neuronList):
            raise RuntimeError('Expected Length:', len(self.neuronList), " recieved length:", len(valueList)  )
        else:
            for i in range(0, len(self.neuronList)):
                self.neuronList[i].setValue(valueList[i])

    def randomizeLayer(self):
        for n in self.neuronList:
            n.randomize()

    def calcLayer(self):                # Make every Neuron in this layer calculate its new value.
        for n in self.neuronList:
            n.calcValue()

class Net(object):
    def __init__(self):
        self.inputLayer = Layer()
        self.outputLayer = Layer()
        self.hiddenLayers = []
        self.hiddenLayerCount = 0

    def addInputLayer(self, listOfValues):
        self.inputLayer.initNeurons(len(listOfValues))
        self.inputLayer.setValues(listOfValues)

    def setInputLayer(self, listOfValues):
        self.inputLayer.setValues(listOfValues)

    def addHiddenLayer(self):
        if len(self.hiddenLayers) == 0:
            self.hiddenLayers.append(Layer(self.inputLayer, self.inputLayer.count))
            self.hiddenLayerCount += 1
        else:
            self.hiddenLayers.append(Layer(self.hiddenLayers[self.hiddenLayerCount-1], self.hiddenLayers[self.hiddenLayerCount-1].count))
            self.hiddenLayerCount += 1

    def addOutputLayer(self, outputNeuronCount):
        if self.hiddenLayerCount == 0:
            self.outputLayer.initNeurons(outputNeuronCount)
            self.outputLayer.connectFully(self.inputLayer)
        else:
            self.outputLayer.initNeurons(outputNeuronCount)
            self.outputLayer.connectFully(self.hiddenLayers[self.hiddenLayerCount-1])

    def randomizeNet(self):
        self.inputLayer.randomizeLayer()
        for layer in self.hiddenLayers:
            layer.randomizeLayer()
        self.outputLayer.randomizeLayer()

    def calcNet(self):
        for layer in self.hiddenLayers:
            layer.calcLayer()
        self.outputLayer.calcLayer()

    def displayOutputWeights(self):
        for i in range(0, len(self.outputLayer.neuronList)):
            print(i," has weight: " ,self.outputLayer.neuronList[i].value)

    def displayNet(self):

        print(len(self.inputLayer.neuronList) , "--", end="")
        for i in self.inputLayer.neuronList:
            print( "[",i.value,"]" , end="")
            for affector in i.affectors:
                print( "a:" , affector )
            print ("")

        print(len(self.hiddenLayers[0].neuronList) , "--", end="")
        for i in self.hiddenLayers[0].neuronList:
            print(  "[",i.value,"]", end="")
            for affector in i.affectors:
                print(  "a:" , affector)
            print ("")

        print(len(self.outputLayer.neuronList), "--", end="")
        for i in self.outputLayer.neuronList:
            print(  "[",i.value,"]", end="")
            for affector in i.affectors:
                print(  "a:" , affector )
            print ("")
