import numpy as np
import torch
import random
import os

np.random.seed(0)
torch.manual_seed(0)

length = 1000
num_samples = 100

def sine_2(X, signal_freq=60.):
    return (np.sin(2 * np.pi * (X) / signal_freq) + np.sin(4 * np.pi * (X) / signal_freq)) / 2.0

def noisy(Y, noise_range=(-0.05, 0.05)):
    noise = np.random.uniform(noise_range[0], noise_range[1], size=Y.shape)
    return Y + noise

def noisy_sine(length, num_samples):
    x = []
    for i in range(num_samples):
        random_offset = random.randint(0, length)
        X = np.arange(length)
        s = sine_2(X + random_offset)
        Y = noisy(s)
        x.append(sample(length))
    x = np.array(x)
    data = x.astype('float64')
    if not os.path.exists('./data'):
        os.mkdir('./data')
    torch.save(data, open('./data/traindata_noisy.pt', 'wb'))
    print("Training Data saved in ./data/traindata_noisy.pt")
    return x

def plane_sine(length, num_samples):
    np.random.seed(2)
    T = 20
    x = np.empty((num_samples, length), 'int64')
    x[:] = np.array(range(length)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')
    if not os.path.exists('./data'):
        os.mkdir('./data')
    torch.save(data, open('traindata_plain.pt', 'wb'))
    print("Training Data saved in ./data/traindata_plain.pt")
    return x
