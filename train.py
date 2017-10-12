from torchvision import transforms
from torch.optim import RMSprop
from PIL import Image


from datasets import NYUDepth
from models import HourGlass
from criterion import RelativeDepthLoss
from train_utils import fit, prep_img, save_checkpoint

from config import config


path = config['data_path']
lr = config['lr']
nb_epoch = config['nb_epoch']
batch_size = config['batch_size']


train = NYUDepth(path+'train', path+'labels_train.pkl', transforms=transforms.ToTensor())
hourglass = HourGlass()
hourglass.cuda()
optimizer = RMSprop(hourglass.parameters(), lr)
criterion = RelativeDepthLoss()

history = fit(hourglass, train, criterion, optimizer, batch_size, nb_epoch)

