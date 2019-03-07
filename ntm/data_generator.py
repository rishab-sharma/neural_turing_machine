import numpy as np
import torch
import random
import os

np.random.seed(0)
torch.manual_seed(0)

class datagen():

    def __init__(self, length, num_samples):
        self.length = length
        self.num_samples = num_samples

    def sine_2(self, X, signal_freq=60.):
        return (np.sin(2 * np.pi * (X) / signal_freq) + np.sin(4 * np.pi * (X) / signal_freq)) / 2.0

    def noisy(self, Y, noise_range=(-0.05, 0.05)):
        noise = np.random.uniform(noise_range[0], noise_range[1], size=Y.shape)
        return Y + noise

    def noisy_sine(self):
        x = []
        for i in range(self.num_samples):
            random_offset = random.randint(0, self.length)
            X = np.arange(self.length)
            s = self.sine_2(X + random_offset)
            Y = self.noisy(s)
            x.append(Y)
        x = np.array(x)
        data = x.astype('float64')
        cwd = os.getcwd()
        if not os.path.exists('./data'):
            os.mkdir('./data')
        torch.save(data, open(cwd+'/traindata_noisy.pt', 'wb'))
        print("Training Data saved in {}/traindata_noisy.pt".format(cwd))
        return data

    def plane_sine(self):
        np.random.seed(2)
        T = 20
        x = np.empty((self.num_samples, self.length), 'int64')
        x[:] = np.array(range(self.length)) + np.random.randint(-4 * T, 4 * T, self.num_samples).reshape(self.num_samples, 1)
        data = np.sin(x / 1.0 / T).astype('float64')
        if not os.path.exists('./data'):
            os.mkdir('./data')
        cwd = os.getcwd()
        torch.save(data, open(cwd+'traindata_plain.pt', 'wb'))
        print("Training Data saved in {}/traindata_plain.pt".format(cwd))
        return data
