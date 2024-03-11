from model import MyLeNet
import numpy as np
import os
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

if __name__ == '__main__':

    # hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256

    # divide train and test
    train_dataset = mnist.MNIST(root='../data/MNIST/train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='../data/MNIST/test', train=False, transform=ToTensor())
    
    # loada data
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # define model
    model = MyLeNet().to(device)
    sgd = SGD(model.parameters(), lr=1e-1)
    loss_fn = CrossEntropyLoss()
    all_epoch = 100
    prev_acc = 0
    epoch_count = 0

    # train, 1 epoch at a time
    for current_epoch in range(all_epoch):
        model.train()
        print(f"epoch {epoch_count}: training")
        for idx, (train_x, train_label) in tqdm(enumerate(train_loader)):
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            
            # reset gradient
            sgd.zero_grad()
            # forward
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            # backward
            loss.backward()
            # update parameters of net
            sgd.step()

            # other possible situations
            # pytorch 在反向传播前为什么要把梯度清零？
            # https://www.zhihu.com/question/303070254

        # evaluate

        all_correct_num = 0
        all_sample_num = 0

        model.eval()
        print(f"epoch {epoch_count}: evaluating")
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

        # save model
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(model, 'models/mnist_{:.3f}.pkl'.format(acc))
        if np.abs(acc - prev_acc) < 1e-4:
            break
        prev_acc = acc

        epoch_count = epoch_count + 1
    
    print("Model finished training")