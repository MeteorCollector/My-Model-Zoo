from model import MyInceptionNet
from dataset import data_loader
import torch
import glob
from tqdm import tqdm
from torch.autograd import Variable

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 32

    model = MyInceptionNet().to(device)
    print(f"using device: {device}")

    # load pkl
    files = glob.glob('./models/inception_*.pkl')
    max_file = max(files, key=lambda x: float(x.split('_')[-1][:-4]))
    with open(max_file, 'rb') as f:
        model = torch.load(max_file)
    print("Loaded pkl from:", max_file)

    test_loader = data_loader(data_dir='../data/CIFAR100',
                              batch_size=batch_size,
                              test=True)
    
    # evaluate
    all_correct_num = 0
    all_sample_num = 0

    model.eval()

    for idx, (images, labels) in tqdm(enumerate(test_loader)):

        # load data
        images = Variable(images).to(device)
        label = Variable(label).to(device)

        # inference
        predict_y = model(images)
        _, predicted = torch.max(predict_y, 1)

        # append result
        all_sample_num += label.size(0)
        current_correct_num = (predicted == label.data).sum()
        all_correct_num += current_correct_num

    # calculate accuracy
    acc = all_correct_num / all_sample_num
    print('accuracy: {:.3f}'.format(acc), flush=True)
