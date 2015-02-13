# Implements a NeuralNet to Classify images of handwritten numbers.
# Michael Riley Feb 2015 -

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

numberOfImagesToRead = 3  # how many images to load. (max 60000 b/c there are only that many in the file.)

labels = open("data/train-labels.idx1-ubyte", 'rb')
images = open("data/train-images.idx3-ubyte", 'rb')

# Set the offsets in the file. Each file describes how many images/labels are in it but for now I only want 500.
labels.seek(8)  # first label (1 byte)
images.seek(16)  # first pixel of first image (1 byte)

#Setup neural net (28*28 = 784)
ImageList = []
LabelList = []

#Print results
while numberOfImagesToRead > 0:
    currentLabel = int.from_bytes(labels.read(1), 'big')
    currentImage = images.read(28 * 28)
    numberOfImagesToRead -= 1

    ImageList.append(currentImage)
    LabelList.append(currentLabel)

for image in ImageList:
    printImage(image)

#Make neuron class.
#Instance some neurons, with 784 incoming connections.
#For each image apply it to the neural net's inputs and find an answer.
# adjust the network slightly to make it more accurate.
# repeat



# Close files and notify console we are finished.
labels.close()
images.close()
print("Done")
