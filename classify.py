print("Hello")

numberOfImagesToRead = 1

labels = open("data/train-labels.idx1-ubyte", 'rb')
images = open("data/train-images.idx3-ubyte", 'rb')

#Set the offsets in the file. Each file describes how many images/labels are in it but for now I only want 500.
labels.seek(8)      # first label (1 byte)
images.seek(16)     # first pixel of first image (1 byte)

#Setup neural net (28*28 = 784)





while numberOfImagesToRead > 0:
    currentLabel = int.from_bytes(labels.read(1), 'big')
    currentImage = images.read(28*28)
    numberOfImagesToRead -= 1


    print(currentLabel)
    bytesPerLine = 28
    for b in currentImage:
        print(b)
        

# Close files and notify console we are finished.
labels.close()
images.close()
print("Done")
