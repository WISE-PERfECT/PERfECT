'''
01/06/2022
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import time
import os
import utils
import numpy as np
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
t = datetime.fromtimestamp(time.time())

device_filename = 'oect_210112_3A.xlsx'
#device_filename = 'oect_210112_3B.xlsx'

# THIS LINE SHOULD BE CHANGED EVERY TIME
save_dir_name = 'emg_0strain_non_resize'
save_dir_name = save_dir_name + '_' + datetime.strftime(t, '%m%d')

#CODES_DIR = 'D:\\Proj\\2021\\211009_RCpower\\oect_RC_simulation'
# dataset path
DATAROOT = os.path.join(os.getcwd(), 'dataset')

# oect data path
DEVICE_DIR = os.path.join(os.getcwd(), 'data')
device_path = os.path.join(DEVICE_DIR, device_filename)
# old path
#old_path = os.path.join(DEVICE_DIR, old_filename)
SAVE_PATH = 'results'
save_dir_name = os.path.join(SAVE_PATH, save_dir_name)

for p in [DATAROOT, SAVE_PATH, save_dir_name]:
    if not os.path.exists(p):
        os.mkdir(p)

# sampling = 8
sampling = 0
num_pulse = 4   # 5
# z_num_pulse = 4
num_pixels = 4 * 16
# new_img_width = int(np.ceil(num_pixels / num_pulse))
new_img_width = 16
batchsize = 1
device_tested_number = 12
# device_tested_number = 4
device_tested_number = 2

digital = False

#batchsize = 128
#te_batchsize = 128

batchsize = 20
te_batchsize = 20

oect_tested_number = 4

# device = torch.device('cuda:0')
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

TRAIN_PATH = os.path.join(DATAROOT, 'EMGdatasetS.pt')
TEST_PATH = os.path.join(DATAROOT, 'EMGdatasetS.pt')

# tr_dataset = utils.SimpleDataset(TRAIN_PATH, num_pulse=num_pulse, crop=False, sampling=sampling)
# te_dataset = utils.SimpleDataset(TEST_PATH, num_pulse=num_pulse, crop=False, sampling=sampling)

tr_dataset = utils.SimpleDatasetEMG(TRAIN_PATH, num_pulse=num_pulse, crop=False, sampling=sampling)
te_dataset = utils.SimpleDatasetEMG(TEST_PATH, num_pulse=num_pulse, crop=False, sampling=sampling)

train_loader = DataLoader(tr_dataset,
                          batch_size=batchsize,
                          shuffle=True)
test_dataloader = DataLoader(te_dataset, batch_size=batchsize)

num_epoch = 40
learning_rate = 1e-2

num_data = len(tr_dataset)
num_te_data = len(te_dataset)

criterion = nn.CrossEntropyLoss()

model = torch.nn.Sequential(
    # nn.Linear(84, 10)
    #nn.Linear(new_img_width, 10)
    nn.Linear(new_img_width, 5)
)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
device_output = utils.oect_data_proc(path=device_path,
                                    device_test_cnt=device_tested_number,
                                    device_read_times=None)
if digital:
    d_outputs = np.arange(2 ** num_pulse) / (2 ** num_pulse - 1)
    device_output = device_output[1]
    device_output[:] = np.arange(2 ** num_pulse)
else:
    # load oect device data

    # oect_data = utils.oect_data_proc_04(old_path, device_tested_number=oect_tested_number)
    # 0-1 normalization
    device_output = (device_output - device_output.min().min()) / (device_output.max().max() - device_output.min().min())

start_time = time.time()
acc_list = []
loss_list = []
for epoch in range(num_epoch):

    acc = []
    loss = 0
    for i, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        this_batch_size = len(data)

        oect_output = utils.batch_rc_feat_extract(data,
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

        # if i_batch % 300 == 0:
        #     print('%d data trained' % i_batch)
    scheduler.step()
    acc_epoch = (sum(acc) * batchsize / num_data).numpy()
    acc_list.append(acc_epoch)
    loss_list.append(loss)

    epoch_end_time = time.time()
    if epoch == 0:
        epoch_time = epoch_end_time - start_time
    else:
        epoch_time = epoch_end_time - epoch_start_time
    epoch_start_time = epoch_end_time
    print("epoch: %d, loss: %.2f, acc: %.6f, time: %.2f" % (epoch, loss, acc_epoch, epoch_time))

te_accs = []
te_outputs = []
targets = []
with torch.no_grad():
    for i, (data, target) in enumerate(test_dataloader):

        this_batch_size = len(data)
        oect_output = utils.batch_rc_feat_extract(data,
                                                  device_output,
                                                  device_tested_number,
                                                  num_pulse,
                                                  this_batch_size)
        output = model(oect_output)
        te_outputs.append(output)
        acc = torch.sum(output.argmax(dim=-1) == target) / te_batchsize
        te_accs.append(acc)
        targets.append(target)
    te_acc = (sum(te_accs) * te_batchsize / num_te_data).numpy()
    print("test acc: %.6f" % te_acc)

    te_outputs = torch.cat(te_outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    # confusion matrix
    conf_mat = confusion_matrix(targets, torch.argmax(te_outputs, dim=1))

    conf_mat_dataframe = pd.DataFrame(conf_mat,
                                      index=list(range(3)),
                                      columns=list(range(3)))

    conf_mat_normalized = conf_mat_dataframe.divide(conf_mat_dataframe.sum(axis=1), axis=0)
    # print("conf_mat: %.6f" % conf_mat_normalized)

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