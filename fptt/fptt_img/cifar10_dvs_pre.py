import torch
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS


class_num = 10
dataset_dir ='./data/cifar10_dvs'
batch_size = 16
split_by = 'number'
T = 20
normalization = None

# print('CIFAR10-DVS downloadable', CIFAR10DVS.downloadable())
# print('resource, url, md5/n', CIFAR10DVS.resource_url_md5())


train_data_loader = torch.utils.data.DataLoader(
    dataset=CIFAR10DVS(dataset_dir, train=True, use_frame=True, frames_num=T,
                            split_by=split_by, normalization=normalization),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True,
    pin_memory=True)
test_data_loader = torch.utils.data.DataLoader(
    dataset=CIFAR10DVS(dataset_dir, train=False, use_frame=True, frames_num=T,
                            split_by=split_by, normalization=normalization),
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    pin_memory=True)

# img,labels = next(iter(train_data_loader))
# img.shape,labels.shape
for i ,(data, target) in enumerate(test_data_loader):
    if i == 2: 
        print(data.shape,target.shape)
        break