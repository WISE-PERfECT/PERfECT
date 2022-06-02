from PIL.Image import new
from numpy.core.fromnumeric import squeeze
import pandas
import pandas as pd
import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def oect_data_proc_04(path, device_tested_number):
    '''
    for April data processing
    '''
    device_excel = pandas.read_excel(path, converters={'pulse': str})

    device_excel['pulse']

    device_data = device_excel.iloc[:, 1:device_tested_number+1]
    device_data.iloc[30] = 0
    device_data = device_data
    ind = device_excel['pulse']
    ind = [str(i).split('‘')[-1].split('’')[-1].split('\'')[-1] for i in ind]
    device_data.index = ind

    return device_data


def oect_data_proc(path, device_test_cnt, num_pulse=5, device_read_times=None):
    '''
    for 0507 data processing
    '''
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


def binarize_dataset(data, threshold):
    data = torch.where(data > threshold * data.max(), 1, 0)
    return data


def reshape(data, num_pulse):
    num_data, h, w = data.shape
    # TODO
    new_data = []
    for i in range(int(w / num_pulse)):
        new_data.append(data[:, :, i * num_pulse: (i+1) * num_pulse])

    new_data = torch.cat(new_data, dim=1)
    return new_data


class SimpleDataset(torch.utils.data.Dataset):
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
        super(SimpleDataset, self).__init__()

        self.get_ori_img = ori_img

        if type(path) is str:
            self.data, self.label = torch.load(path)
        elif type(path) is tuple:
            self.data, self.label = path[0], path[1]
        else:
            print('wrong path type')
        # except:
        #     self.data, self.label = path['']
        self.ori_img = self.data

        if crop:
            # self.data = self.data[:, 4: 26, 5: 25]
            self.data = self.data[:, 5: 25, 5: 25]
        if sampling != 0:
            self.data = self.data.unsqueeze(dim=1)
            self.data = F.interpolate(self.data, size=(sampling, sampling))
            self.data = self.data.squeeze()

        self.new_img = self.data

        # plt.figure()
        # plt.imshow(self.data[0])
        # plt.savefig('downsampled_img')

        num_data = self.data.shape[0]
        img_h, img_w = self.data.shape[1], self.data.shape[2]
        num_pixel = img_h * img_w
        # self.data = img_h
        if type(path) is str:
            self.data = binarize_dataset(self.data, threshold=0.25)
            # self.ori_img = self.data

            # if num_pixel % num_pulse == 0:
            #     self.data = self.data.view((num_data, num_pulse, -1))
            # else:
            #     self.data = self.data
            self.data = reshape(self.data, num_pulse)
            self.data = torch.transpose(self.data, dim0=1, dim1=2)

        else:
            self.data = torch.squeeze(self.data)
            self.label = torch.squeeze(self.label)
        if transform is not None:
            self.transform = transform

    def __getitem__(self, index):

        # label = self.label_dict[img_name]
        target = self.label[index]

        img = self.data[index]
        if self.get_ori_img:
            # ori_img: original MNIST data
            return img, self.ori_img[index], self.new_img[index], target
        else:
            return img, target

    def __len__(self):
        return self.data.shape[0]

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

        # except:
        #     self.data, self.label = path['']
        self.ori_img = self.data
        self.new_img = self.data

        # plt.figure()
        # plt.imshow(self.data[0])
        # plt.savefig('downsampled_img')

        num_data = self.data.shape[0]
        img_h, img_w = self.data.shape[1], self.data.shape[2]
        num_pixel = img_h * img_w
        # self.data = img_h
        if type(path) is str:
            self.data = binarize_dataset(self.data, threshold=0.25)
            # self.ori_img = self.data

            # if num_pixel % num_pulse == 0:
            #     self.data = self.data.view((num_data, num_pulse, -1))
            # else:
            #     self.data = self.data
            #self.data = reshape(self.data, num_pulse)
            self.data = torch.transpose(self.data, dim0=1, dim1=2)

        else:
            self.data = torch.squeeze(self.data)
            self.label = torch.squeeze(self.label)
        if transform is not None:
            self.transform = transform

    def __getitem__(self, index):

        # label = self.label_dict[img_name]
        target = self.label[index]

        img = self.data[index]
        if self.get_ori_img:
            # ori_img: original MNIST data
            return img, self.ori_img[index], self.new_img[index], target
        else:
            return img, target

    def __len__(self):
        return self.data.shape[0]


# def img_reshape(data, num_pulse, padding=False):
#     new_im = []
#     img_width = data.shape[-1]


def rc_feature_extraction(data, device_data, device_tested_number, num_pulse, padding=False):
    '''
    use device to extract feature (randomly select a experimental output value corresponding to the input binary digits)
    :param data: input data
    :param device_data: experimental device output. a dataframe with 5bit digits as index.
    :param img_width:
    :param device_tested_number: how many bits used for simulating
    :return:
    '''
    device_outputs = []
    img_width = data.shape[-1]
    for i in range(img_width):
        # random index of device outputs
        rand_ind = np.random.randint(1, device_tested_number + 1)
        # binary ind of image data
        # ind = np.array2string(data[:, :, i].numpy()).split('[')[-1].split(']')[0].split(' ')
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
    return device_outputs


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
        features.append(feature)
    features = torch.cat(features, dim=0)
    return features


class ImagesForDemo():
    def __init__(self, path) -> None:
        self.ori_images = []
        self.new_images = []
        self.reshaped_data = []
        self.probabilites = []
        self.targets = []
        self.path = path
        if os.path.exists(path) is not True:
            os.mkdir(path)

    def update_images(self, data, img, new_img, target, output):

        p = F.softmax(output, dim=1).max()
        if output.argmax(dim=-1) == target:
            if target not in self.targets:
                self.ori_images.append(img)
                self.reshaped_data.append(data)
                self.new_images.append(new_img)
                self.targets.append(target)
                self.probabilites.append(p)
            else:
                idx = self.targets.index(target)
                if p > self.probabilites[idx]:
                    self.probabilites[idx] = p
                    self.ori_images[idx] = img
                    self.new_images[idx] = new_img
                    self.reshaped_data[idx] = data

    def save_images(self):
        for i, target in enumerate(self.targets):
            target = target.tolist()[0]
            cls_path = os.path.join(self.path, str(target))
            if os.path.exists(cls_path) is not True:
                os.mkdir(cls_path)
            image_name = f'image_confidence_{int(self.probabilites[i] * 10000): d}.jpg'
            image_name = os.path.join(cls_path, image_name)
            cropped_img_name = os.path.join(cls_path, 'cropped_image')
            pulse_name = os.path.join(cls_path, 'pulses')

            cv2.imwrite(image_name, self.ori_images[i].squeeze().numpy())
            # cv2.imwrite(pulse_name, self.reshaped_data[i].squeeze().numpy())
            plt.figure()
            plt.imshow(self.reshaped_data[i].squeeze())
            plt.savefig(pulse_name)
            plt.close()

            # save cropped img with plt
            plt.figure()
            plt.imshow(self.new_images[i].squeeze())
            plt.savefig(cropped_img_name)
            plt.close()
