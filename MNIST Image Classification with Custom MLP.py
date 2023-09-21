#Name: Kranti Adsul
#USC ID: 5747-0737-86
#Homework 5

#Question 2b

import h5py
import math
import numpy as np
import matplotlib.pyplot as plt

file = 'mnist_traindata.hdf5'
with h5py.File(file, 'r') as cols:
    xdata = np.float32(np.array(cols['xdata']))
    ydata = np.float32(np.array(cols['ydata']))

X_train = xdata[0:50000, :]
Y_train = ydata[0:50000, :]
print('Dimensions of xdata: ', X_train.shape)
print('Dimensions of ydata: ', Y_train.shape)
X_val = xdata[50000:, :]
Y_val = ydata[50000:, :]
print('Dimensions of xdata: ', X_val.shape)
print('Dimensions of ydata: ', Y_val.shape)

# ReLU function
def ReLU(x):
    return np.maximum(x, 0)

def ReLU_cond(x):
    return (x > 0) * 1

# tanh function
def tanh(x):
    return np.tanh(x)

def tanh_cond(x):
    return 1 - np.tanh(x)**2

# softmax function
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)

def training(wt1, wt2, b1, b2, batchsize, lr_rate, x, y, ReLU_is=True):
    sample, data = x.shape
    WT1 = np.zeros([wt1.shape[0], wt1.shape[1]])
    WT2 = np.zeros([wt2.shape[0], wt2.shape[1]])
    B1 = np.zeros([b1.shape[0], ])
    B2 = np.zeros([b2.shape[0], ])
    acc = 0

    for i in range(sample):
        sample_pt = x[i]
        if ReLU_is:
            relu_1 = ReLU(np.matmul(wt1, sample_pt) + b1)
        else:
            relu_1 = tanh(np.matmul(wt1, sample_pt) + b1)
        relu_2 = softmax(np.matmul(wt2, relu_1) + b2)

        if np.argmax(relu_2) == np.argmax(y[i]):
            acc += 1

        D2 = relu_2 - y[i]
        B2 += D2
        WT2 = WT2 + np.outer(D2, relu_1.T)
        D1 = np.matmul(wt2.T, D2) * ReLU_cond(relu_1)
        B1 += D1
        WT1 = WT1 + np.outer(D1, sample_pt.T)

        if (i + 1) % batchsize == 0:
            wt1 -= lr_rate * (WT1 / batchsize)
            wt2 -= lr_rate * (WT2 / batchsize)
            b1 -= lr_rate * (B1 / batchsize)
            b2 -= lr_rate * (B2 / batchsize)
            WT1 = np.zeros([wt1.shape[0], wt1.shape[1]])
            WT2 = np.zeros([wt2.shape[0], wt2.shape[1]])
            B1 = np.zeros([b1.shape[0], ])
            B2 = np.zeros([b2.shape[0], ])

    return wt1, wt2, b1, b2, acc / sample


def forward_prop(w1, w2, b1, b2, x, y, is_ReLU=True):
    samples, data = x.shape
    acc = 0
    
    for i in range(samples):
        eachx = x[i]
        if is_ReLU:
            relu1 = ReLU(np.matmul(w1, eachx) + b1)
        else:
            relu1 = tanh(np.matmul(w1, eachx) + b1)

        relu2 = softmax(np.matmul(w2, relu1) + b2)

        if np.argmax(y[i]) == np.argmax(relu2):
            acc += 1

    return acc / samples

#Code for ReLU activation function

batch = 200
lrs = [0.06]
epochs = 50

# Define initial weights and biases
wt1 = np.random.uniform(-0.01, 0.01, [200, 784])
wt2 = np.random.uniform(-0.01, 0.01, [10, 200])
b1 = np.random.uniform(-0.01, 0.01,[200,])
b2 = np.random.uniform(-0.01, 0.01,[10,])

train_accs, test_acc = {}, {}
for lr in lrs:
    # Reset weights and biases for each learning rate
    wt1_trial = wt1.copy()
    wt2_trial = wt2.copy()
    b1_trial = b1.copy()
    b2_trial = b2.copy()
    
    lr_trial = lr
    train_accs[lr] = []
    test_acc[lr] = []
    
    for i in range(epochs):
        # Train for one epoch
        wt1_trial, wt2_trial, b1_trial, b2_trial, train_acc = training(wt1_trial, wt2_trial, b1_trial, b2_trial, batch, lr_trial, X_train, Y_train, True)
        
        # Calculate validation accuracy
        val_acc = forward_prop(wt1_trial, wt2_trial, b1_trial, b2_trial, X_val, Y_val, True)

        
        # Divide the learning rate by 2 after the 6th and 18th epochs
        if i == 6 or i == 18:
            lr_trial /= 2
        
        train_accs[lr].append(train_acc)
        test_acc[lr].append(val_acc)
        
    # Print testing and training accuracy for the last epoch
    print("Training accuracy for learning rate", lr, "with ReLU activation function: ", train_accs[lr][-1])
    print("Test accuracy for learning rate", lr, "with ReLU activation function: ", test_acc[lr][-1])

    # Plot the training and validation accuracy curves
    plt.plot(range(epochs), train_accs[lr], label=f"Training (lr={lr})")
    plt.plot(range(epochs), test_acc[lr], label=f"Validation (lr={lr})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"ReLU activation function with Learning rate={lr}")
    plt.axvline(x=6, color='r', linestyle='--')
    plt.axvline(x=18, color='r', linestyle='--')
    plt.legend()
    plt.show()
