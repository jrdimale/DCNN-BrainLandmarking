# CODE TO TRAIN A MODEL FOR MULTIPLE LANDMARKS
# A PyTorch implementation of a neural network for automatic landmark detection.
# The model is trained on a set of images and their corresponding landmarks.
# Author: Jordi Male (jordi.male@salle.url.edu)
        
import torch
import torch.nn as nn
import cv2
import numpy as np
import math
import sys

from torch.autograd import Variable
from torch.optim import Adam
from skimage import exposure
from sklearn.model_selection import train_test_split

# Number of landmarks to be detected
NUM_LANDMARKS = 8

# File containing the names of the images to be used for training
TRAIN_IMAGES_LIST = "xxx/xxx.csv"

# File containing the names of the images to be used for testing
TEST_IMAGES_LIST = "xxx/xxx.csv"

# Directories containing the images and landmarks  to be used for training
TRAIN_IMAGE_DIRECTORY = "/xxx/xxx/"
TRAIN_LANDMARKS_DIRECTORY = "/xxx/xxx/"

# Directory containing the images and landmarks to be used for testing
TEST_IMAGE_DIRECTORY = "/xxx/xxx/"
TEST_LANDMARKS_DIRECTORY = "/xxx/xxx/"

# Name of the model to be saved
MODEL_NAME = "model_name.model"

# Name of the checkpoint to be saved
CHECKPOINT_NAME = "checkpoint_name.pt"

# Reference image for histogram matching
REFERENCE_IMAGE = "xxx/xxx.png"

# File extensions for path formatting
IMAGE_EXTENSION = ".png"
LANDMARK_EXTENSION = ".txt"

# Train - test split size
TEST_SIZE = 0.3

# DCNN parameters
LEARNING_RATE = 0.005
EPOCHS = 100000
BATCH_SIZE = 256
GAMMA = 0.95

# Image parameters
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

BASE_X = 50
BASE_Y = 50
END_X = 200
END_Y = 200

# TODO Change accordingly to the GPU being used
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Define if training is resumed from a checkpoint
RESUME_CHECKPOINT = False

# Epochs per checkpoint
EPOCHS_PER_CHECKPOINT = 50

class MultiLandmarkModel(nn.Module):
    """
    A PyTorch neural network model for automatic landmark detection.
    """
    def __init__(self):
        super(MultiLandmarkModel, self).__init__()

        # Define the convolutional layers for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Define the fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(81*32*32, 1000),  # Adjust the input size according to your needs
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(500, NUM_LANDMARKS * 2) # Output layer with 16 classes, to predict a 8 two-dimensional coordinates
        )

        # Initialize the weights and biases of convolutional and linear layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        # Forward pass through the network
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

class EuclideanDistanceLoss(nn.Module):
    """
    Custom loss function to calculate Euclidean distance loss for landmark prediction.
    """
    def __init__(self, num_landmarks):
        super(EuclideanDistanceLoss, self).__init__()
        self.num_landmarks = num_landmarks

    def forward(self, preds, targets):
        loss = 0
        for i in range(self.num_landmarks):
            x_pred, y_pred = preds[:, 2*i], preds[:, 2*i+1]
            x_target, y_target = targets[:, 2*i], targets[:, 2*i+1]
            landmark_loss = torch.sqrt(torch.pow(x_target - x_pred, 2) + torch.pow(y_target - y_pred, 2))
            loss += landmark_loss
        
        loss = loss / self.num_landmarks

        return loss.mean()

def normalize_batch(batch):
    """
    Normalize a batch of images.

    Args:
        batch: Batch of images to be normalized

    Returns:
        batch: Normalized batch of images
    """
    mean = batch.mean()
    std = batch.std()
    return (batch - mean) / std



def load_data(IMAGES_LIST, IMAGE_DIRECTORY, LANDMARKS_DIRECTORY):
    """
    Load and preprocess training data
    Can be used for both training and testing of a single landmark model.

    Args:
        IMAGES_LIST: File containing the names of the images to be used for training/testing
        IMAGE_DIRECTORY: Directory containing the images to be used for training/testing
        LANDMARKS_DIRECTORY: Directory containing the landmarks to be used for training/testing
    
    Returns:
        images_tensor: Tensor containing the images
        landmarks_tensor: Tensor containing the landmarks
    """

    # Change the path accordingly, for test or train data
    images_file = open(IMAGES_LIST, 'r')
    
    list_of_images = images_file.readlines()

    landmarks_list = []
    images_list = []

    reference_image = cv2.imread(REFERENCE_IMAGE, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.resize(reference_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)

    # Crop the images so its size is 150x150
    cropped_reference_image = reference_image[BASE_Y:END_Y, BASE_X:END_X]

    for image_name in list_of_images:

        # ---- LOAD THE IMAGE ----
        image_path = IMAGE_DIRECTORY + image_name + IMAGE_EXTENSION

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # The image is not preprocessed, it could be preprocessed here (resize, flip, etc)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)

        # Crop the images so its size is 150x150
        cropped_image = image[BASE_Y:END_Y, BASE_X:END_X]
        
        # Perform histogram matching
        cropped_image_matched = exposure.match_histograms(cropped_image, cropped_reference_image, multichannel=False)

        images_list.append(cropped_image_matched)

        # ---- LOAD THE LANDMARKS ----
        # Path formatting
        landmarks_path = LANDMARKS_DIRECTORY + image_name + LANDMARK_EXTENSION
        landmarks_file = open(landmarks_path, 'r')

        landmarks = landmarks_file.readlines()
        image_landmarks_list = []

        for landmark in landmarks:

            # TODO Change according to the format of the landmarks file
            coord_x = landmark[0]
            coord_y = landmark[1]

            coord_x = float(coord_x)
            coord_y = float(coord_y)

            image_landmarks_list.append(coord_x - BASE_X)
            image_landmarks_list.append(coord_y - BASE_Y)
        
        landmarks_list.append(image_landmarks_list)

       
    print("[INFO] Total data loaded: {} images and their N anatomical landmarks".format(len(images_list)))

    # Convert the images and landmarks into PyTorch tensors
    images_tensor = torch.stack(images_list)
    landmarks_tensor = torch.stack(landmarks_list)

    
    return images_tensor, landmarks_tensor

def train_model():
    """
    Train the multi landmark model.
    """

    training_images, training_landmarks = load_data(TRAIN_IMAGES_LIST, TRAIN_IMAGE_DIRECTORY, TRAIN_LANDMARKS_DIRECTORY)

    # Split the data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(training_images, training_landmarks, test_size=TEST_SIZE)

    # Print the sizes of the training and validation sets
    print(f"Training set size: {len(x_train)}")
    print(f"Validation set size: {len(x_val)}")
    
    model = MultiLandmarkModel().to(device)

    # print(model)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08)

    criterion = EuclideanDistanceLoss(NUM_LANDMARKS)

    train_losses = []
    validation_losses = []

    x_train, y_train = Variable(x_train), Variable(y_train)
    x_val, y_val = Variable(x_val), Variable(y_val)

    num_batches_train = math.ceil(len(x_train) / BATCH_SIZE)
    num_batches_val = math.ceil(len(x_val) / BATCH_SIZE)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

    start_epoch = 0

    # Resume training from a checkpoint - If speficied
    if RESUME_CHECKPOINT:
        checkpoint = torch.load(CHECKPOINT_NAME)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        validation_losses = checkpoint['validation_losses']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        optimizer.param_groups[0]['lr'] = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']

        print(f"Resuming training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, EPOCHS):

        torch.cuda.empty_cache()

        model.train()

        for batch_idx in range(num_batches_train):

            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(x_train))
            x_batch = x_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]

            
            # Train the model on the current batch
            optimizer.zero_grad()

            x_batch = normalize_batch(x_batch)

            outputs = model(x_batch.to(device))

            loss_train = criterion(outputs, y_batch.to(device))
            loss_train.backward()
            optimizer.step()

            train_losses.append(loss_train.item())
        
        model.eval()

        # Evaluate the model on the validation set after each epoch
        with torch.no_grad():

            for batch_idx in range(num_batches_val):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(x_val))
                x_batch = x_val[start_idx:end_idx]
                y_batch = y_val[start_idx:end_idx]

                # Evaluate the model on the validation batch
                x_batch = normalize_batch(x_batch)
                val_outputs = model(x_batch.to(device))
                val_loss = criterion(val_outputs, y_batch.to(device))
                validation_losses.append(val_loss)
            
        # Print the current epoch, training loss, validation loss and learning rate
        print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {validation_losses[-1]:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
        
        # Save the model and optimizer state after each 50 epochs
        if (epoch + 1) % EPOCHS_PER_CHECKPOINT == 0:
            checkpoint_name = CHECKPOINT_NAME
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'validation_losses': validation_losses,
                # Add any other additional information you want to save
            }, checkpoint_name)

            print(f"Checkpoint saved as {checkpoint_name}")
    
    # Save the model after training
    torch.save(model.state_dict(), MODEL_NAME)

def test_model():
    """
    Test the multi-landmark model.
    """
    
    testing_images, testing_landmarks = load_data(TEST_IMAGES_LIST, TEST_IMAGE_DIRECTORY, TEST_LANDMARKS_DIRECTORY)

    testing_model = MultiLandmarkModel().to(device)

    print(f"Testing model: {MODEL_NAME}")

    checkpoint = torch.load(MODEL_NAME, map_location=torch.device('cpu'))
    testing_model.load_state_dict(checkpoint['model_state_dict'])

    testing_model.eval()
    testing_outputs = []

    # Evaluate the model on the testing set
    for testing_image in testing_images:
        testing_image = testing_image.unsqueeze(0)
        testing_image = normalize_batch(testing_image)

        testing_image = testing_image.to(device)
        testing_output = testing_model(testing_image)
        testing_outputs.append(testing_output)
    
    # TODO Compute the desired error for testing set in a single landmark model
    # mean_error = compute_error(testing_outputs, testing_landmarks)

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train_model()
    elif sys.argv[1] == 'test':
        test_model()
    
