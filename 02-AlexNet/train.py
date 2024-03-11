from model import MyAlexNet
import numpy as np
import os
import torch
from torchvision.datasets import CIFAR10
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm
import glob

if __name__ == '__main__':

    # hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"using device: {device}")
    batch_size = 500

    # preprocess image
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # divide train and test
    train_dataset = CIFAR10(root='../data/CIFAR10', train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root='../data/CIFAR10', train=False, transform=transform)
    
    # loada data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # config data
    alex_config = \
    {
        'lr': 1e-3,                   # learning rate
        'l2_regularization':1e-4,     # L2 regularization efficient
        'num_classes': 10,
    }

    # define model
    model = MyAlexNet(alex_config).to(device)

    # load pkl from historical ones
    files = glob.glob('./models/alex_*.pkl')
    max_file = max(files, key=lambda x: float(x.split('_')[-1][:-4]))
    with open(max_file, 'rb') as f:
        model = torch.load(max_file)
    print("Loaded pkl from:", max_file)

    opt = torch.optim.Adam(model.parameters(), lr=alex_config['lr'], weight_decay=alex_config['l2_regularization'])
    loss_fn = CrossEntropyLoss()
    all_epoch = 20
    prev_acc = 0
    epoch_count = 0

    # train, 1 epoch at a time
    for current_epoch in range(all_epoch):
        model.train()
        print(f"epoch {epoch_count}: training")
        for idx, (images, labels) in tqdm(enumerate(train_loader)):
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            
            # reset gradient
            opt.zero_grad()
            # forward
            predict_y = model(images)
            loss = loss_fn(predict_y, labels)
            # backward
            loss.backward()
            # update parameters of net
            opt.step()

        # evaluate
        all_correct_num = 0
        all_sample_num = 0

        model.eval()
        print(f"epoch {epoch_count}: evaluating")
        for idx, (images, labels) in tqdm(enumerate(test_loader)):

            # load data
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            # inference
            predict_y = model(images)
            _, predicted = torch.max(predict_y, 1)

            # append result
            all_sample_num += labels.size(0)
            current_correct_num = (predicted == labels.data).sum()
            all_correct_num += current_correct_num

        # calculate accuracy
        acc = all_correct_num / all_sample_num
        print('accuracy: {:.3f}'.format(acc), flush=True)

        # save model
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(model, 'models/alex_{:.3f}.pkl'.format(acc))
        if np.abs((acc - prev_acc).cpu()) < 1e-4:
            break
        prev_acc = acc

        epoch_count = epoch_count + 1
    
    print("Model finished training")