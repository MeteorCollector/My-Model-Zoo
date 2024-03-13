from model import MyInceptionNet
from dataset import data_loader
import numpy as np
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from tqdm import tqdm
import glob

if __name__ == '__main__':

    # hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"using device: {device}")
    
    num_classes = 100
    batch_size = 16
    learning_rate = 0.003
    weight_decay = 0.005

    # CIFAR100 dataset 
    train_loader, valid_loader = data_loader(data_dir='../data/CIFAR100',
                                         batch_size=batch_size)


    # define model
    model = MyInceptionNet(num_classes).to(device)

    # load pkl from historical ones
    files = glob.glob('./models/inception_*.pkl')
    if files:
        max_file = max(files, key=lambda x: float(x.split('_')[-1][:-4]))
        with open(max_file, 'rb') as f:
            model = torch.load(max_file)
        print("Loaded pkl from:", max_file)

    opt = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
        for idx, (images, labels) in tqdm(enumerate(valid_loader)):

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
        torch.save(model, 'models/inception_{:.3f}.pkl'.format(acc))
        if np.abs((acc - prev_acc).cpu()) < 1e-4:
            break
        prev_acc = acc

        epoch_count = epoch_count + 1
    
    print("Model finished training")