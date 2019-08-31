# Imports here
# Xiaogang He from China
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
import seaborn as sns
from sklearn.preprocessing import normalize
from collections import OrderedDict
import argparse


#LOAD CHECKPOINT
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    #Load pre-trained network
    if model == 'vgg16':
        fc_model = models.vgg16(pretrained=True)
    elif model == 'densenet121':
        fc_model = models.densenet121(pretrained=True)
    elif model == 'alexnet':
        fc_model = models.alexnet(pretrained = True)
   
    fc_model.eval()

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_size'][0])),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(checkpoint['hidden_size'][0],checkpoint['hidden_size'][1])),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(checkpoint['hidden_size'][1], checkpoint['output_size'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    fc_model.classifier = classifier
    fc_model.load_state_dict(checkpoint['state_dict'])
    
    optimizer = optim.Adam(fc_model.classifier.parameters())
    optimizer.state_dict = checkpoint['optimizer']
    
    fc_model.mapping = checkpoint['mapping']
    fc_model.epochs = checkpoint['epochs']
    
    
    return fc_model, optimizer

#PROCESS an IMAGE
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    size = 256, 256
    img.thumbnail(size)
    
    width, height = img.size
    img = img.crop(((width//2 - 224//2, height//2 - 224//2, width//2 + 224//2, height//2 + 224//2)))
    
    numpy_image = np.array(img)/255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    normal_img = []
    
    for i in numpy_image:
         normal_img.append((i-mean)/std)
    
    normal_img = np.asarray(normal_img)
    
    tensor_img = torch.from_numpy(normal_img.transpose(2,0,1))
    
    return tensor_img

#PREDICT CATEGORY
def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    
    model.to(device)
    
    tensor_image = process_image(image_path)
    tensor_image = tensor_image.unsqueeze_(0)
    tensor_image = tensor_image.to(device, dtype=torch.float)
    
    logps = model(tensor_image)
    ps = torch.exp(logps)
    
    top_p, top_class = ps.topk(topk, dim=1)
    
    #find right class in dictionary
    flower_class = []
    name_class = []
    top_class = np.squeeze(np.asarray(top_class))   
    for item in top_class:
        flower_class.append(list(new_model.mapping.keys())[list(new_model.mapping.values()).index(item)])
    
    for t in flower_class:
        name_class.append(cat_to_name[t])
    
    return top_p, flower_class, name_class