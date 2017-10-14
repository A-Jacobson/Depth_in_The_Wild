import argparse
from torchvision import transforms
from torch.optim import RMSprop
import torch
import matplotlib.pyplot as plt

from datasets import NYUDepth
from models import HourGlass
from criterion import RelativeDepthLoss
from train_utils import fit, save_checkpoint
from torch.backends import cudnn


def main(data_path, label_path, nb_epoch, save_path,
         start_path=None, batch_size=2, lr=1e-3, plot_history=True):
    cudnn.benchmark = True
    train = NYUDepth(data_path, data_path, transforms=transforms.ToTensor())
    hourglass = HourGlass()
    hourglass.cuda()
    optimizer = RMSprop(hourglass.parameters(), lr)

    if start_path:
        experiment = torch.load(start_path)
        hourglass.load_state_dict(experiment['model_state'])
        optimizer.load_state_dict(experiment['optimizer_state'])
    criterion = RelativeDepthLoss()

    history = fit(hourglass, train, criterion, optimizer, batch_size, nb_epoch)
    save_checkpoint(hourglass.state_dict(), optimizer.state_dict(), save_path)
    if plot_history:
        plt.plot(history['loss'], label='loss')
        plt.xlabel('epoch')
        plt.ylabel('relative depth loss')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', default='/home/austin/data/NYU/train')
    parser.add_argument('label_path', default='/home/austin/data/NYU/labels_train.pkl')
    parser.add_argument('nb_epoch')
    parser.add_argument('save_path')
    parser.add_argument('start_path', default=None)
    parser.add_argument('batch_size', default=2)
    parser.add_argument('lr', default=1e-3)
