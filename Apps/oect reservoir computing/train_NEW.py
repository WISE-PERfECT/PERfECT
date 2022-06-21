'''
June 21, 2022
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# hyperparameters
sampling = 0
num_pulse = 4
num_pixels = 4 * 16
img_width = 16
device_tested_number = 2    # number of bits used for simulating

num_epoch = 40
learning_rate = 1e-2

digital = False

batchsize = 20
te_batchsize = 20


# binarize the data with threshold = 0.25
def binarize_dataset(data, threshold):
    data = torch.where(data > threshold * data.max(), 1, 0)
    return data


class SimpleDatasetEMG(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 num_pulse,
                 crop=False,
                 transform=None,
                 sampling=0,
                 ori_img=False):

        '''
        ori_img: if return the original MNIST img and crop&resized img
        '''

        super(SimpleDatasetEMG, self).__init__()

        self.get_ori_img = ori_img

        if type(path) is str:
            self.data, self.label = torch.load(path)

        self.ori_img = self.data
        self.new_img = self.data
        self.num_data = self.data.shape[0]

        if type(path) is str:
            self.data = binarize_dataset(self.data, threshold=0.25)
            self.data = torch.transpose(self.data, dim0=1, dim1=2)

        else:
            self.data = torch.squeeze(self.data)
            self.label = torch.squeeze(self.label)
        if transform is not None:
            self.transform = transform

    def __getitem__(self, index):
        target = self.label[index]
        img = self.data[index]
        if self.get_ori_img:
            return img, self.ori_img[index], self.new_img[index], target
        else:
            return img, target


    def __len__(self):
        return self.data.shape[0]


def rc_feature_extraction(data, device_data, device_tested_number, num_pulse, padding=False):
    '''
    use device to extract feature (randomly select a experimental output value corresponding to the input binary digits)
    :param data: input data
    :param device_data: experimental device output. a dataframe with 5bit digits as index.
    :param device_tested_number: how many bits used for simulating
    :return: (batch_size, img_width)
    '''
    device_outputs = []
    img_width = data.shape[-1]
    for i in range(img_width):
        # random index of device outputs
        rand_ind = np.random.randint(1, device_tested_number + 1)

        if len(data.shape) == 3:
            ind = [str(idx) for idx in data[0, :, i].numpy()]
        elif len(data.shape) == 2:
            ind = [str(idx) for idx in data[:, i].numpy()]

        ind = ''.join(ind)

        if num_pulse == 4 and padding:
            ind = '1' + ind
            output = device_data.loc[ind, rand_ind]
        elif device_tested_number in [2, 4, 5]:
            output = device_data.loc[ind, rand_ind]
        elif device_tested_number == 1:
            output = device_data.loc[ind]

        device_outputs.append(output)

    device_outputs = torch.unsqueeze(torch.tensor(device_outputs, dtype=torch.float), dim=0)

    return device_outputs                                       # shape: (1, img_width)


def batch_rc_feat_extract(data,
                          device_output,
                          device_tested_number,
                          num_pulse,
                          batch_size):
    features = []
    for batch in range(batch_size):
        single_data = data[batch]
        feature = rc_feature_extraction(single_data,
                                        device_output,
                                        device_tested_number,
                                        num_pulse)
        features.append(feature)                                # shape: (1, img_width)
    features = torch.cat(features, dim=0)
    return features                                             # shape: (batch_size, img_width)


# Input: path:              the path of oect tabel: 'oect_210112_3A.xlsx'
#        device_test_cnt:   how many bits used for simulating
#        num_pulse:         
def oect_data_proc(path, device_test_cnt, num_pulse = 4, device_read_times=None):
    device_excel = pd.read_excel(path, converters={'pulse': str})

    device_read_time_list = ['10s', '10.5s', '11s', '11.5s', '12s']
    if device_read_times == None:
        cnt = 0
    else:
        cnt = device_read_time_list.index(device_read_times)

    num_rows = 2 ** num_pulse
    device_data = device_excel.iloc[cnt * (num_rows + 1): cnt * (num_rows + 1) + num_rows, 0: device_test_cnt + 1]

    # use binary pulse as dataframe index
    device_data.index = device_data['pulse']
    del device_data['pulse']

    return device_data


# Function: develop the path of training set, testing set, device reading path
# Output:   TRAIN_PATH: path of training set
#           TEST_PATH: path of testing set
#           device_path: device reading path
#           save_dir_name: name of the directory results saved in
def set_path():
    t = datetime.fromtimestamp(time.time())

    # THIS LINE SHOULD BE CHANGED EVERY TIME
    save_dir_name = 'emg_0strain_non_resize'
    save_dir_name = save_dir_name + '_' + datetime.strftime(t, '%m%d')

    device_filename = 'oect_210112_3A.xlsx'

    # dataset path
    DATAROOT = os.path.join(os.getcwd(), 'dataset')

    # oect data path
    DEVICE_DIR = os.path.join(os.getcwd(), 'data')
    device_path = os.path.join(DEVICE_DIR, device_filename)

    # result path
    SAVE_PATH = 'results'
    save_dir_name = os.path.join(SAVE_PATH, save_dir_name)

    for p in [DATAROOT, SAVE_PATH, save_dir_name]:
        if not os.path.exists(p):
            os.mkdir(p)

    TRAIN_PATH = os.path.join(DATAROOT, 'EMGdatasetS.pt')
    TEST_PATH = os.path.join(DATAROOT, 'EMGdatasetS.pt')

    return TRAIN_PATH, TEST_PATH, device_path, save_dir_name


TRAIN_PATH, TEST_PATH, device_path, save_dir_name = set_path()
tr_dataset = SimpleDatasetEMG(TRAIN_PATH, num_pulse=num_pulse, sampling=sampling)
te_dataset = SimpleDatasetEMG(TEST_PATH, num_pulse=num_pulse, sampling=sampling)

train_loader = DataLoader(tr_dataset,
                        batch_size=batchsize,
                        shuffle=True)
test_dataloader = DataLoader(te_dataset, batch_size=batchsize)

criterion = nn.CrossEntropyLoss()

model = torch.nn.Sequential(
    nn.Linear(img_width, 5)
)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
device_output = oect_data_proc(path=device_path,
                                    device_test_cnt=device_tested_number,
                                    device_read_times=None)

if digital:
    device_output = device_output[1]
    device_output[:] = np.arange(2 ** num_pulse)
else:
    # 0-1 normalization
    device_output = (device_output - device_output.min().min()) / (device_output.max().max() - device_output.min().min())


# Start training
start_time = time.time()

acc_list = []
loss_list = []
for epoch in range(num_epoch):
    acc = []
    loss = 0
    for i, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        this_batch_size = len(data)

        oect_output = batch_rc_feat_extract(data,
                                                  device_output,
                                                  device_tested_number,
                                                  num_pulse,
                                                  this_batch_size)

        logic = model(oect_output)
        batch_loss = criterion(logic, target)
        loss += batch_loss
        batch_acc = torch.sum(logic.argmax(dim=-1) == target) / batchsize
        acc.append(batch_acc)
        batch_loss.backward()
        optimizer.step()

    scheduler.step()
    acc_epoch = (sum(acc) * batchsize / tr_dataset.num_data).numpy()
    acc_list.append(acc_epoch)
    loss_list.append(loss)

    epoch_end_time = time.time()
    if epoch == 0:
        epoch_time = epoch_end_time - start_time
    else:
        epoch_time = epoch_end_time - epoch_start_time
    epoch_start_time = epoch_end_time
    print("epoch: %d, loss: %.2f, acc: %.6f, time: %.2f" % (epoch, loss, acc_epoch, epoch_time))


# testing
te_accs = []
te_outputs = []
targets = []
with torch.no_grad():
    for i, (data, target) in enumerate(test_dataloader):

        this_batch_size = len(data)
        oect_output = batch_rc_feat_extract(data,
                                                  device_output,
                                                  device_tested_number,
                                                  num_pulse,
                                                  this_batch_size)
        output = model(oect_output)
        te_outputs.append(output)
        acc = torch.sum(output.argmax(dim=-1) == target) / te_batchsize
        te_accs.append(acc)
        targets.append(target)
    te_acc = (sum(te_accs) * te_batchsize / te_dataset.num_data).numpy()
    print("test acc: %.6f" % te_acc)

    te_outputs = torch.cat(te_outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    # confusion matrix
    conf_mat = confusion_matrix(targets, torch.argmax(te_outputs, dim=1))

    conf_mat_dataframe = pd.DataFrame(conf_mat,
                                      index=list(range(3)),
                                      columns=list(range(3)))

    conf_mat_normalized = conf_mat_dataframe.divide(conf_mat_dataframe.sum(axis=1), axis=0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_mat_dataframe, annot=True, fmt='d')
    plt.savefig(os.path.join(save_dir_name, 'conf_mat'))
    plt.close()
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_mat_normalized, annot=True)
    plt.savefig(os.path.join(save_dir_name, 'conf_mat_normlized'))
    plt.close()

    torch.save(model, os.path.join(save_dir_name, 'downsampled_img_mode.pt'))


plt.figure()
plt.plot(acc_list)
plt.show()
