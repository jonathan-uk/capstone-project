import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FashionCNN(nn.Module):
    
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out


def cnnnet():
    model = FashionCNN()
    model.to(device)
    return model


def model_fn(model_dir):
    print("In model_fn. Model directory is -")
    print(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnnnet().to(device)
    
    with open(os.path.join(model_dir, "model.pt"), "rb") as f:
        print("Loading the dog-classifier model")
        checkpoint = torch.load(f , map_location =device)
        model.load_state_dict(checkpoint)
        print('MODEL-LOADED')
        logger.info('model loaded successfully')
    model.eval()
    return model


def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    #if content_type == JPEG_CONTENT_TYPE: return io.BytesIO(request_body)
    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    logger.debug(f'Request body TYPE is: {type(request_body)}')
    if content_type == JPEG_CONTENT_TYPE: return Image.open(io.BytesIO(request_body))
    logger.debug('SO loded JPEG content')
    # process a URL submitted to the endpoint
    
    if content_type == JSON_CONTENT_TYPE:
        #img_request = requests.get(url)
        logger.debug(f'Request body is: {request_body}')
        request = json.loads(request_body)
        logger.debug(f'Loaded JSON object: {request}')
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# inference
def predict_fn(input_object, model):
    logger.info('In predict fn')

    mean = [0.2860819389520745]
    std = [0.3529394901810539]

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    logger.info("transforming input")
    input_object=test_transform(input_object)
    with torch.no_grad():
        logger.info("Calling model")
        prediction = model(input_object.unsqueeze(0))

    return prediction.cpu().numpy().tolist()
