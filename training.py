#Get train and test loaders from our local file 'preprocessing'.
from preprocessing import train_test_loaders

from torch.utils.data import Dataset 
from torch.autograd import Variable 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torch 

'''
Visit this link to see how I built my pytorch neural net! It's an amazing tutorial.
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
This is a more specific one to my case 
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html 
'''
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # max pooling over a (2,2) window 
        self.pool = nn.MaxPool2d(2,2)
        '''
        Define three 2D convolutional layers 
        First layer: 1 input image channel, 6 output channels, 3x3 square convolution
        etc...
        '''
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.conv3 = nn.Conv2d(6, 16, 3)
        #affline operation y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 from image dimension
        self.fc2 = nn.Linear(120, 48) 
        self.fc3 = nn.Linear(48, 24)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def train(net, criterion, optimizer, trainloader, epoch):
    loss = 0.0
    for i, data in enumerate(trainloader, 0):
        #Convert the dataframe to tensor not str
        inputs = Variable(data['image'].float())
        labels = Variable(data['label'].long())

        # Zero the parameter gradients 
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels[:, 0])
        loss.backward()
        optimizer.step()

        # show loss/epoch
        loss += loss.item()
        if i % 100 == 0:
            print('epoch: {} ============ loss: {}'.format(epoch, loss / (i+1)))
    print("Finished Training Current Epoch")

def main():
    '''
    Initializing the network. Takes in the model as a float then applies the loss function,
    optimizer, then saves the result to a checkpoint
    '''
    net = Model().float()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_load, test_load = train_test_loaders()
    # Here we can define the number of epochs
    for epoch in range(12): 
        train(net, criterion, optimizer, train_load, epoch)
        scheduler.step()
    torch.save(net.state_dict(), "checkpoint.pth")

if __name__ == '__main__':
    main()