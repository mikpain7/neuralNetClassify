from distutils.core import setup

setup(
    name='neuralNetClassify',
    version='',
    packages=[''],
    url='',
    license='',
    author='PhilSkins',
    author_email='',
    description=''
)
import neuralNet
net = neuralNet.Net()
net.setInputLayer([0.5,0.1,0.1])
net.addHiddenLayer()
net.addOutputLayer(1)
net.randomizeNet()
net.displayNet()