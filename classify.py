# Implements a NeuralNet to Classify images of handwritten numbers.
# Michael Riley Feb 2015 -

import neuralNet

def printImage(currentImageToPrint):
    charsPerLine = 28       # Use to configure the numbers of chars output per line.
    charactersPrinted = 0
    for b in currentImageToPrint:
        charactersPrinted += 1
        if b > 0:
            print("8", end='')
        else:
            print("-", end='')
        if charactersPrinted == 28:
            charactersPrinted = 0
            print("")

def initNeuralNetwork(filename=None):
    if filename==None: # start from scratch....
        #Init Neural Net.
        net = neuralNet.Net()
        net.addInputLayer(currentList)   # start with first image for dimensions.
        net.addHiddenLayer()
        net.addHiddenLayer()

        net.addOutputLayer(10)          # one for each digit.. also want to add an extra for garbled/unable to determine.
        net.randomizeNet()
        return net
    else:
        file = open(filename, 'rb')
        #net = pickle.load(file)
        file.close()

def saveNeuralNetwork(net, filename=None):
    if filename==None:
        raise RuntimeError('Can\'t open file without filename.')
    else:
        file = open(filename, 'wb')
       # pickle.dump(net, file, -1)
        file.close()


numberOfImagesToRead = 1000  # how many images to load. (max 60000 b/c there are only that many in the file.)

labels = open("data/train-labels.idx1-ubyte", 'rb')
images = open("data/train-images.idx3-ubyte", 'rb')

# Set the offsets in the file. Each file describes how many images/labels are in it but for now I only want 500.
labels.seek(8)  # first label (1 byte)
images.seek(16)  # first pixel of first image (1 byte)

#Setup neural net (28*28 = 784)
ImageList = []
LabelList = []
ImageListNormalized = []

#Print results
while numberOfImagesToRead > 0:
    currentLabel = int.from_bytes(labels.read(1), 'big')
    currentImage = images.read(28 * 28)
    numberOfImagesToRead -= 1

    ImageList.append(currentImage)
    LabelList.append(currentLabel)

    currentList = []
    for char in currentImage:
        currentList.append(char.__int__()/255)
    ImageListNormalized.append(currentList)

net = initNeuralNetwork('network1')
net = initNeuralNetwork()

numberCorrectSoFar = 0

for i in range(0, len(ImageListNormalized)):
    print("Image number:", i+1, " it should be identified as a:", LabelList[i])
    #printImage(ImageListNormalized[i])
    net.setInputLayer(ImageListNormalized[i])
    net.calcNet()
    net.calcGradientsNet(neuralNet.makeTargetVector(LabelList[i]))

    net.applyCalculatedGradientsToNet()
    net.displayBestGuess()
    if net.returnBestGuess() == LabelList[i]:
        numberCorrectSoFar += 1
    print("Correct: " + str(numberCorrectSoFar) + "  Percentage: " + str(numberCorrectSoFar/(i+1)))

    print("----------------------------------------------------------------")
    #net.displayOutputWeights()

saveNeuralNetwork(net, 'network1')      # this is currently not working.
# Close files and notify console we are finished.
labels.close()
images.close()
print("Done")
