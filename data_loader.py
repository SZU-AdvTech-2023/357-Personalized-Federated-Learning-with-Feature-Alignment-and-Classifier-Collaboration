import numpy as np
from numpy.core.fromnumeric import trace
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import pdb
import os
import glob
from shutil import copyfile
import json


# CIFAR
def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(2022)
    num_classes = len(np.unique(dataset.targets))
    shard_per_user = num_classes
    imgs_per_shard = int(len(dataset) / (num_users * shard_per_user))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs_dict = {}
    for i in range(len(dataset)):
        label = dataset.targets[i]
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)
    
    rand_set_all = []
    if len(rand_set_all) == 0:
        for i in range(num_users):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            # pdb.set_trace()
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users[i] = np.concatenate(rand_set)

    for key, value in dict_users.items():
        assert(len(np.unique(torch.tensor(dataset.targets)[value]))) == shard_per_user

    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR dataset
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(2022)
    shard_per_user = 3
    imgs_per_shard = int(len(dataset) / (num_users * shard_per_user))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    idxs_dict = {}
    for i in range(len(dataset)):
        label = dataset.targets[i]
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    rand_set_all = []
    if len(rand_set_all) == 0:
        for i in range(num_users):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            # pdb.set_trace()
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users[i] = np.concatenate(rand_set)

    for key, value in dict_users.items():
        assert(len(np.unique(torch.tensor(dataset.targets)[value]))) == shard_per_user

    return dict_users


def cifar_noniid_s(dataset, num_users, noniid_s=20, local_size=600, train=True):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(2022)
    s = noniid_s/100
    num_per_user = local_size if train else 300
    num_classes = len(np.unique(dataset.targets))

    noniid_labels_list = [[0,1,2], [2,3,4], [4,5,6], [6,7,8], [8,9,0]]

    # -------------------------------------------------------
    # divide the first dataset
    num_imgs_iid = int(num_per_user*s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    num_samples = len(dataset)
    num_per_label_total = int(num_samples/num_classes)
    labels1 = np.array(dataset.targets)
    idxs1 = np.arange(len(dataset.targets))
    # iid labels
    idxs_labels = np.vstack((idxs1, labels1))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # label available
    label_list = [i for i in range(num_classes)]
    # number of imgs has allocated per label
    label_used = [2000 for i in range(num_classes)] if train else [500 for i in range(num_classes)]
    iid_per_label = int(num_imgs_iid/num_classes)
    iid_per_label_last = num_imgs_iid - (num_classes-1)*iid_per_label

    np.random.seed(2022)
    for i in range(num_users):
        # allocate iid idxs
        label_cnt = 0
        for y in label_list:
            label_cnt = label_cnt + 1
            iid_num = iid_per_label
            start = y*num_per_label_total+label_used[y]
            if label_cnt == num_classes:
                iid_num = iid_per_label_last
            if (label_used[y]+iid_num)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+iid_num]), axis=0)
            label_used[y] = label_used[y] + iid_num
        # allocate noniid idxs
        # rand_label = np.random.choice(label_list, 3, replace=False)
        rand_label = noniid_labels_list[i%5]
        noniid_labels = len(rand_label)
        noniid_per_num = int(num_imgs_noniid/noniid_labels)
        noniid_per_num_last = num_imgs_noniid - noniid_per_num*(noniid_labels-1)
        label_cnt = 0
        for y in rand_label:
            label_cnt = label_cnt + 1
            noniid_num = noniid_per_num
            start = y*num_per_label_total+label_used[y]
            if label_cnt == noniid_labels:
                noniid_num = noniid_per_num_last
            if (label_used[y]+noniid_num)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+noniid_num]), axis=0)
            label_used[y] = label_used[y] + noniid_num
        dict_users[i] = dict_users[i].astype(int)
    return dict_users

## --------------------------------------------------
## loading dataset
## --------------------------------------------------

def cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root='data', train=True, download=True, transform=transform_train)  #
    testset = datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform_test)
    print("CIFAR10 Data Loading...")
    return trainset, testset


class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, index=None):
        self.dataset = dataset
        self.idxs = [int(i) for i in index] if index is not None else [int(i) for i in range(len(dataset))]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    train_dataset = []
    test_dataset = []
    user_groups_train = {}
    user_groups_test = {}
    train_loader = []
    test_loader = []
    global_test_loader = []

    if args.dataset in ['cifar', 'cifar10']:
        train_dataset, test_dataset = cifar10()
        # sample training data amongst users
        if args.iid:
            # Sample IID user data 
            user_groups_train = cifar_iid(train_dataset, args.num_users)
            user_groups_test = cifar_iid(test_dataset, args.num_users)
            print('IID Data Loading---')
        else:
            # Sample Non-IID user data 
            user_groups_train = cifar_noniid_s(train_dataset, args.num_users, args.noniid_s, args.local_size, train=True)
            user_groups_test = cifar_noniid_s(test_dataset, args.num_users, args.noniid_s, args.local_size, train=False)
            print('non-IID Data Loading---')

    else:
        raise NotImplementedError()

    return train_loader, test_loader, global_test_loader


if __name__ =='__main__':
    pass
