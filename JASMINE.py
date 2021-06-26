import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, H, Dx, Dy, D):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(Dx, H)
        self.fc2 = nn.Linear(Dy, H)
        self.fc3 = nn.Linear(H,H)
        self.fc4 = nn.Linear(H, D)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x)+self.fc2(y))
        h2 = F.relu(self.fc3(h1))
        h3 = self.fc4(h2)
        return h3

def Jasmine(x,y,n_epoch = 500, H = 100, data_proportion = 0.2, window_size=0.1, Plotting = True):
    Dx = x.shape[1]
    Dy = y.shape[1]
    data_size = int(data_proportion*x.shape[0])
    sx = normalize(x)
    sy = normalize(y)
    xy = np.concatenate((sx,sy), axis=1)
    model = Net(H, Dx, Dy, D)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    plot_loss = []
    for epoch in tqdm(range(n_epoch)):
        xy = xy[np.random.permutation(xy.shape[0]), :]
        x_sample=xy[:data_size,:x.shape[1]]
        y_sample=xy[:data_size,x.shape[1]:]
        y_shuffle = np.random.permutation(y_sample)

        x_sample = Variable(torch.from_numpy(x_sample).type(torch.FloatTensor), requires_grad = True)
        y_sample = Variable(torch.from_numpy(y_sample).type(torch.FloatTensor), requires_grad = True)
        y_shuffle = Variable(torch.from_numpy(y_shuffle).type(torch.FloatTensor), requires_grad = True)

        pred_xy = model(x_sample, y_sample)
        pred_x_y = model(x_sample, y_shuffle)

        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = - ret
        plot_loss.append(loss.data.numpy())
        model.zero_grad()
        loss.backward()
        optimizer.step()
    pl = -np.array(plot_loss)
    res = signal.savgol_filter(pl, int(data_size*window_size) + 1, 3)
    if Plotting == True:
        plt.plot(np.arange(n_epoch)+1, res)
        plt.xlabel("epoch, n")
        plt.ylabel("Mutual information, nat")
        plt.grid(True)
        plt.show()
    print("\n Mutual information is ", res[-1])
    return res[-1]

def normalize(z):
    sz = np.zeros(z.shape)
    for i in range(z.shape[1]):
        sz[:,i] = (z[:,i] - np.mean(z[:,i]))/np.std(z[:,i])
    return sz