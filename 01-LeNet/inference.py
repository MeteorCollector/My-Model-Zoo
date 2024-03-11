from model import MyLeNet
import numpy as np
import os
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import glob
from tqdm import tqdm

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    model = MyLeNet().to(device)

    # load pkl
    files = glob.glob('./models/mnist_*.pkl')
    max_file = max(files, key=lambda x: float(x.split('_')[-1][:-4]))
    with open(max_file, 'rb') as f:
        model = torch.load(max_file)
    print("Loaded pkl from:", max_file)

    test_dataset = mnist.MNIST(root='../data/MNIST/test', train=False, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model.eval()

    all_correct_num = 0
    all_sample_num = 0
        
    for idx, (test_x, test_label) in tqdm(enumerate(test_loader)):

        # load data
        test_x = test_x.to(device)
        test_label = test_label.to(device)

        # inference
        predict_y = model(test_x.float()).detach()
        predict_y =torch.argmax(predict_y, dim=-1)

        # append result
        current_correct_num = predict_y == test_label
        all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
        all_sample_num += current_correct_num.shape[0]

    # calculate accuracy
    acc = all_correct_num / all_sample_num
    print('accuracy: {:.3f}'.format(acc), flush=True)
