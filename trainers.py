from collections import defaultdict
from shutil import copyfile

import torch
from tqdm import tqdm_notebook
from torch.utils.data import DataLoader
from torch.autograd import Variable


class Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = None
        self.criterion = None
        self.epoch_pbar = None
        self.total_pbar = None
        self.loss_meter = None
        self.history = defaultdict(list)

    def compile(self, optimizer, criterion):
        self.optimizer = optimizer
        self.criterion = criterion

    def forward(self, data, target):
        data = Variable(data.cuda())
        target = Variable(target.cuda())
        output = self.model(data)
        loss = self.criterion(output, target)
        return output, loss

    def fit_iteration(self, data, target):
        output, loss = self.forward(data, target)
        self.loss_meter.update(loss.data[0])
        self.history['loss'].append(loss.data[0])
        self.epoch_pbar.set_description("[ loss: {:.4f} ]".format(self.loss_meter.avg))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fit_epoch(self, loader):
        self.model.train()
        self.total_pbar = tqdm_notebook(loader, total=len(loader))
        for data, target in self.total_pbar:
            self.fit_iteration(data, target)

    def fit(self, train, nb_epoch=1, batch_size=32, shuffle=True,
            validation_data=None, cuda=True, num_workers=0):
        if validation_data:
            print('Train on {} samples, Validate on {} samples'.format(len(train), len(validation_data)))
        else:
            print('Train on {} samples'.format(len(train)))
        train_loader = DataLoader(train, batch_size, shuffle, num_workers=num_workers, pin_memory=True)
        self.total_pbar = tqdm_notebook(range(nb_epoch), total=nb_epoch)
        for _ in self.total_pbar:
            self.fit_epoch(train_loader)
            if validation_data:
                 self.validate(validation_data, batch_size)
