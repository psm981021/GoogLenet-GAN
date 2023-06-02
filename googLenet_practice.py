import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from collections import OrderedDict
import torch.optim as optim




import torchvision.models as models
model = models.googlenet(pretrained = True)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
for params in model.parameters():
    #print(params)
    params.requires_grad = False 

#setting the model parameters to fix the data
model.fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024,2048)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(2048,2)),
    ('output', nn.LogSoftmax(dim = 1))
    ]))
#print(model)

#dataloader function

def load_data(data_folder, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    
    data = torchvision.datasets.ImageFolder(root = data_folder, transform = transform)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    return data_loader 

data_folder = '/Users/parkseongbeom/pytorch-test/deeplearning_practice/cat vs dog/train'
batch_size = 32
num_workers = 0
dataloader = load_data(data_folder, batch_size, num_workers)

"""#visualization

random_batch = random.choice(list(dataloader))
samples, labels = random_batch

# Generate random indices for images in the batch
num_images = 5  # Number of images to visualize
random_indices = random.sample(range(samples.shape[0]), num_images)

# Visualize the random images
fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

for i, idx in enumerate(random_indices):
    image = samples[idx].numpy().transpose((1, 2, 0))
    label = labels[idx].item()

    axes[i].imshow(image)
    axes[i].set_title(f"Label: {label}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()"""
import torchvision.models as models
model = models.googlenet(pretrained = True)

# model part
model = model.to(device) #shifting model to gpu
loss = nn.CrossEntropyLoss()
loss1 = nn.CrossEntropyLoss()
loss2 = nn.CrossEntropyLoss()
discount = 0.3
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001, amsgrad=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500], gamma=0.5)

epochs = 3
itr = 1
p_itr = 200
model.train()
total_loss = 0
loss_list = []
acc_list = []
for epoch in range(epochs):
    for samples, labels in dataloader:
        samples = samples.to(device)
        samples = samples.to(torch.float32) 
        labels = labels.to(device)
        
        #for param in model.parameters():
        #    param.requires_grad = True
        #o,o1,o2 = model(samples) -> googlenet outputs using eager output
        _loss = model(samples)
        
        #check labels
        loss_value = loss(_loss,labels)
        #loss_value = loss(o,labels) + discount*(loss1(o1,labels) + loss2(o2,labels))
        #loss = criterion(output[0], labels)
        loss_value.backward()
        optimizer.step()
        total_loss += loss_value.item()
        scheduler.step() 
        #output = torch.cat([o1, o2, o], dim=1)
        output = total_loss
        
        if itr%p_itr == 0:
            pred = torch.argmax(output, dim=1)
            correct = pred.eq(labels)
            acc = torch.mean(correct.float())
            print('Correct:{} pred:{} labels:[]'.format(correct,pred,labels)) 
            print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr, total_loss/p_itr, acc))
            loss_list.append(total_loss/p_itr)
            acc_list.append(acc)
            total_loss = 0
        
            
        itr += 1

# extract a feature with trained googlenet give it a condition to GAN
# train googlenet to extract features to identify cat and dogs
# with trained features extract inorder to use it as a condition for C-GAN
# to find out whether the models are trained give several inputs and test out whether the training holds and gan actually works

input_image = ''#!!
input_image = transforms(input_image)
input_image = input_image.unsqueeze(0)  # Add batch dimension


num_epochs = 100

feature = model(input_image)


# Convert the extracted features to a tensor
conditional_data = feature.clone().detach()

import torch.nn as nn

# !!기초적인 GAN -> 확인해야함
class Generator(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_size + num_classes, 1024)
        # Add more layers as needed

    def forward(self, features, conditional_data):
        x = torch.cat([features, conditional_data], dim=1)
        x = self.fc(x)
        # Apply additional layers and transformations
        return x

# Define the discriminator architecture
class Discriminator(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(input_size + num_classes, 1)
        # Add more layers as needed

    def forward(self, image, features, conditional_data):
        x = torch.cat([image, features, conditional_data], dim=1)
        x = self.fc(x)
        # Apply additional layers and transformations
        return x
    
    
# !!check whether num_class holds
generator = Generator(input_size=model.fc.out_features, num_classes=2)
discriminator = Discriminator(input_size=model.fc.out_features, num_classes=2)

# !! 차이 확인, 해당 코드는 model feature를 직접적으로 c-gan의 condition으로 사용하려고 한다.
#generator = Generator(input_size=feature.size(1), num_classes=2)
#discriminator = Discriminator(input_size=feature.size(1), num_classes=2)


# Define loss function and optimizer
adversarial_loss = nn.BCEWithLogitsLoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
for epoch in range(num_epochs):
    for real_images, _ in dataloader:  # Replace 'dataloader' with your actual data loading mechanism
        batch_size = real_images.size(0)

        # Generate fake images using the generator
        fake_images = generator(feature, conditional_data)
        
        # Train the discriminator
        discriminator_real_outputs = discriminator(real_images, feature, conditional_data)
        discriminator_fake_outputs = discriminator(fake_images.detach(), feature, conditional_data)
        
        discriminator_real_loss = adversarial_loss(discriminator_real_outputs, torch.ones(batch_size, 1))
        discriminator_fake_loss = adversarial_loss(discriminator_fake_outputs, torch.zeros(batch_size, 1))
        discriminator_loss = discriminator_real_loss + discriminator_fake_loss

        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train the generator
        discriminator_fake_outputs = discriminator(fake_images, feature, conditional_data)
        generator_loss = adversarial_loss(discriminator_fake_outputs, torch.ones(batch_size, 1))

        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()
"""
With these changes, CGAN will now be conditioned on the features extracted from the trained GoogLeNet model. 
The generator takes the features as input and generates fake images, 
while the discriminator receives both real and fake images along with the corresponding features as input for discrimination.

"""