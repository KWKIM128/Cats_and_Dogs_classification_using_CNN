"""
Date: 1st January, 2023

Cats and dogs classification using CNN
"""

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn as nn
import visdom
from data_loader import CatsDogs


# visdom
vis = visdom.Visdom()

# Checks for GPU
torch.cuda.is_available()


# set devicie
device = torch.device('cuda')

# Dataset

    
# Hyperparameters
num_epochs = 40
batch_size = 20
learning_rate = 0.001


# Transforming images
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.RandomRotation(degrees=45),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomCrop((224,224)),
        transforms.ToTensor()
    ]
) 

# Load dataset
dataset = CatsDogs(csv_file='cat_dog.csv', root_dir='cat_dog', 
                   transform=transform)

# Split dataset into training, valdiation, and testing
train_set, test_set = torch.utils.data.random_split(dataset, [22500, 2500])

train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                           batch_size=batch_size, shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                           batch_size=batch_size, shuffle=True)
classes = ('cat', 'dog')

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
print(images[0].shape)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=0,stride=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=0,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,padding=0,stride=1),
            nn.MaxPool2d(2, stride=2)
        
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64,32,kernel_size=3,padding=0,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,padding=0,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,padding=0,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        
        )
        
        
        
        self.fc1 = nn.Linear(32*5*5, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        out =self.layer1(x)
        out =self.layer2(out)
        out =self.layer3(out)
        out =self.layer4(out)
        out =self.layer5(out)
        out =out.view(out.size(0),-1)
        out =self.relu(self.fc1(out))
        out =self.relu(self.fc2(out))
        return out
    
def weights_init_uniform_rule(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)
            
model = ConvNet().to(device)
model.apply(weights_init_uniform_rule)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


print("..........................................")
print("Start Training")
print("..........................................")


vis.line([0.], [0.], win='train_loss', opts=dict(title='train loss', xlabel='Iterations'))

global_step = 0
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    # training loop
    total_train = 0
    for i, (images, labels) in enumerate(train_loader):
        
        images = images.to(device)
        labels = labels.to(device)
        
        # forward pass
        output = model(images)
        loss = criterion(output, labels)
        
        #backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if (i+1) % 225 == 0:
            print(f'epoch: {epoch+1} / {num_epochs}, step: {i+1} / {n_total_steps}, loss = {loss.item():.4f}')
        vis.line([loss.item()], [global_step], win='train_loss', update='append')
        total_train += loss.item()
        
    train_avg = total_train/n_total_steps
    print(f'epoch: {epoch+1} / {num_epochs}, avgerage loss = {train_avg:.4f}')


       
       
print("..........................................")
print("Finished training")


print("..........................................")
print("Start Testing")
print("..........................................")

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(2)]
    n_class_samples = [0 for i in range(2)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(2):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')  

