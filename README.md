# Cats_and_Dogs_classification_using_CNN
Image classification task using simple CNN architecture. 
Dataset downloaded from Kaggle (https://www.kaggle.com/datasets/ashfakyeafi/cat-dog-images-for-classification)

## Average Loss Value Per Epoch

## Mdoel Architecture
Hyperparameters: number of epochs: 10 | batch size: 20 | learning rate: 0.001

Layer (type:depth-idx)                   Output Shape              Param 

├─Sequential: 1-1                        [-1, 64, 110, 110]        --

|    └─Conv2d: 2-1                       [-1, 64, 222, 222]        1,792

|    └─ReLU: 2-2                         [-1, 64, 222, 222]        --

|    └─Conv2d: 2-3                       [-1, 64, 220, 220]        36,928

|    └─ReLU: 2-4                         [-1, 64, 220, 220]        --

|    └─MaxPool2d: 2-5                    [-1, 64, 110, 110]        --

├─Sequential: 1-2                        [-1, 64, 54, 54]          --

|    └─Conv2d: 2-6                       [-1, 64, 108, 108]        36,928

|    └─MaxPool2d: 2-7                    [-1, 64, 54, 54]          --

├─Sequential: 1-3                        [-1, 32, 26, 26]          --

|    └─Conv2d: 2-8                       [-1, 32, 52, 52]          18,464

|    └─ReLU: 2-9                         [-1, 32, 52, 52]          --

|    └─MaxPool2d: 2-10                   [-1, 32, 26, 26]          --

├─Sequential: 1-4                        [-1, 32, 12, 12]          --

|    └─Conv2d: 2-11                      [-1, 32, 24, 24]          9,248

|    └─ReLU: 2-12                        [-1, 32, 24, 24]          --

|    └─MaxPool2d: 2-13                   [-1, 32, 12, 12]          --

├─Sequential: 1-5                        [-1, 32, 5, 5]            --

|    └─Conv2d: 2-14                      [-1, 32, 10, 10]          9,248

|    └─ReLU: 2-15                        [-1, 32, 10, 10]          --

|    └─MaxPool2d: 2-16                   [-1, 32, 5, 5]            --

├─Linear: 1-6                            [-1, 512]                 410,112

├─ReLU: 1-7                              [-1, 512]                 --

├─Linear: 1-8                            [-1, 512]                 262,656

├─ReLU: 1-9                              [-1, 512]                 --

-----------------------------------------------------------------------------

Total params: 785,376

Trainable params: 785,376

Non-trainable params: 0

Total mult-adds (G): 2.36

-----------------------------------------------------------------------------

Input size (MB): 0.57

Forward/backward pass size (MB): 54.23

Params size (MB): 3.00

Estimated Total Size (MB): 57.80

