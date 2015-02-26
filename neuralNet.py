__author__ = 'Michael'
# Class definition for Neural Nets
# There are 3 layers
# -top- neural net (a Collection of layers)
# -mid- layer (a collection of neurons)
# -low- neuron (a collection of links)
import random
import math

#constants
LEARNING_RATE = 0.1        #controls how fast the network responds.

def makeTargetVector(targetNumber):
    tVec = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    tVec[targetNumber] += 0.8
    return tVec

class Neuron(object):
    def __init__(self):
        self.affectors = {}  # contains a link to a neuron and an associated weight.
        self.affectees = []
        self.value = -1  # default out-of bounds value.
        self.bias = 0
        self.gradient = 0

    def addAffector(self, n, weight=None):
        if weight == None:
            weight = random.random()
        self.affectors[n] = weight
        n.affectees.append(self)

    def getWeight(self, n):
        if n in self.affectors.keys():
            return self.affectors[n]  # lookup weight in affectors dictonary.
        if n in self.affectees:
            return n.affectors[self]  # lookup weight in other neuron's affectors dict
        raise RuntimeError("Those Neurons don't seem to be connected. there is no weight between them.")

    def adjustWeightByFactor(self, n,
                             factor):  # adjusts the source neurons wieght by that factor. (0.3 means 30%) used for making the network learn.
        self.affectors[n] = self.affectors[n] * factor

    def setWeight(self, n, weight):
        self.affectors[n] = weight

    def setValue(self, value):
        self.value = value

    def activationFunction(self, value):  # does the mathematics of appling the sigmoid function. (logistic function)
        return (1 / (1 + math.exp(-(0.4*value))))

    def calcValue(self):
        self.value = 0  # this is the accumulator
        for i in self.affectors.keys():
            self.value += i.value * self.affectors[i]  # sum the weights together. with their values.
        self.value = self.value - self.bias
       # print("preactivation value: ", self.value)
        self.value = self.activationFunction(self.value)  # pass this through the activation function defined above.
        #print("postactivation value: ", self.value)
        # print (self.value, self.threshold) # for debug

    def calcGradientForOutputLayer(self, targetOutputValue):
        # for output layer neurons.. different from hidden layer neurons as they depend on the output layers.
        self.gradient = (targetOutputValue - self.value) * (self.activationFunction(self.value) * (1 - self.activationFunction(self.value)))
       # print ("Gradient(outputlayer):", self.gradient)

    def calcGradientForHiddenLayer(self):
        gradWeightSum = 0
        for i in self.affectees:
            gradWeightSum += i.getWeight(self) * i.gradient
        self.gradient = gradWeightSum * (self.activationFunction(self.value) * (1 - self.activationFunction(self.value)))
        #print ("Gradient:", self.gradient)

    def applyGradient(self):
        for n in self.affectors.keys():
            weightDelta = LEARNING_RATE * self.gradient * n.value
            self.affectors[n] += weightDelta
            biasDelta = LEARNING_RATE * self.gradient
            self.bias =+ biasDelta

    def randomize(self):  # Randomizes current value and weights associated with each source neuron.
        self.value = random.uniform(-1,1)               # randomize value.. if wieghts are to be used without first calling calcValue
        self.bias = random.uniform(-1,1)
        for i in self.affectors.keys():                 # for every link...
            self.affectors[i] = random.uniform(-1,1)    # assign a value between 0 and 1


class Layer(object):
    def __init__(self, layerBelow=None, layerSize=None):
        if layerBelow == None:
            self.neuronList = []
            self.count = 0
        else:
            self.neuronList = []
            self.count = 0
            self.initNeurons(layerSize)
            self.connectFully(layerBelow)

    def addNeuron(self, neuron):
        self.neuronList.append(neuron)
        self.count += 1

    def initNeurons(self, layerSize):
        for i in range(0, layerSize):
            self.addNeuron(Neuron())

    def connectFully(self, layerBelow):
        for i in self.neuronList:
            for lbNeuron in layerBelow.neuronList:
                i.addAffector(lbNeuron)  # add with random weight.

    def setValues(self, valueList):
        if len(valueList) != len(self.neuronList):
            raise RuntimeError('Expected Length:', len(self.neuronList), " recieved length:", len(valueList))
        else:
            for i in range(0, len(self.neuronList)):
                self.neuronList[i].setValue(valueList[i])

    def randomizeLayer(self):
        for n in self.neuronList:
            n.randomize()

    def calcLayer(self):  # Make every Neuron in this layer calculate its new value.
        for n in self.neuronList:
            n.calcValue()

    def calcGradForOutputLayer(self, targetVector):  # needs to know what the target is.
        if len(targetVector) != len(self.neuronList):
            raise RuntimeError("targetVector is not same length as neuronlist")
        for i in range(0, len(self.neuronList)):
            self.neuronList[i].calcGradientForOutputLayer(targetVector[i])

    def calcGradForHiddenLayer(self):
        for n in self.neuronList:
            n.calcGradientForHiddenLayer()

    def applyCalculatedGradientToLayer(self):
        for n in self.neuronList:
            n.applyGradient()


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
        if self.hiddenLayerCount == 0:
            self.hiddenLayers.append(Layer(self.inputLayer, self.inputLayer.count))
            self.hiddenLayerCount += 1
        else:
            self.hiddenLayers.append(Layer(self.hiddenLayers[self.hiddenLayerCount - 1], self.hiddenLayers[self.hiddenLayerCount - 1].count))
            self.hiddenLayerCount += 1

    def addOutputLayer(self, outputNeuronCount):
        if self.hiddenLayerCount == 0:
            self.outputLayer.initNeurons(outputNeuronCount)
            self.outputLayer.connectFully(self.inputLayer)
        else:
            self.outputLayer.initNeurons(outputNeuronCount)
            self.outputLayer.connectFully(self.hiddenLayers[self.hiddenLayerCount - 1])

    def randomizeNet(self):
        self.inputLayer.randomizeLayer()
        for layer in self.hiddenLayers:
            layer.randomizeLayer()
        self.outputLayer.randomizeLayer()

    def calcNet(self):  # works left to right from input to output.
        for layer in self.hiddenLayers:
            layer.calcLayer()
        self.outputLayer.calcLayer()

    def calcGradientsNet(self, tVec):  # works backwards. from output towards input.
        self.outputLayer.calcGradForOutputLayer(tVec)
        for layer in self.hiddenLayers:
            layer.calcGradForHiddenLayer()

    def applyCalculatedGradientsToNet(self):
        self.outputLayer.applyCalculatedGradientToLayer()
        for layer in self.hiddenLayers:
            layer.applyCalculatedGradientToLayer()

    def displayOutputWeights(self):
        for i in range(0, len(self.outputLayer.neuronList)):
            print("Output Neuron:"+str(i)+ " has weight: "+ str(self.outputLayer.neuronList[i].value), end="")
            if self.outputLayer.neuronList[i].value > 0.9:
                print("<--")
            else:
                print("")

    def displayBestGuess(self):
        highestIndex = self.returnBestGuess()
        print("Best guess, with certainty: " + str(self.outputLayer.neuronList[highestIndex].value) + "  its a: " + str(highestIndex))

    def returnBestGuess(self):
        highestVal = 0
        highestIndex = -1
        for i in range(0, len(self.outputLayer.neuronList)):
            if self.outputLayer.neuronList[i].value > highestVal:
                highestVal = self.outputLayer.neuronList[i].value
                highestIndex = i
        return highestIndex

    def displayNet(self):

        print(len(self.inputLayer.neuronList), "--", end="")
        for i in self.inputLayer.neuronList:
            print("[", i.value, "]", end="")
            for affector in i.affectors:
                print("a:", affector)
            print("")

        print(len(self.hiddenLayers[0].neuronList), "--", end="")
        for i in self.hiddenLayers[0].neuronList:
            print("[", i.value, "]", end="")
            for affector in i.affectors:
                print("a:", affector)
            print("")

        print(len(self.outputLayer.neuronList), "--", end="")
        for i in self.outputLayer.neuronList:
            print("[", i.value, "]", end="")
            for affector in i.affectors:
                print("a:", affector)
            print("")
