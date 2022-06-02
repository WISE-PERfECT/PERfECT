import torch

path = "results/没拉伸/downsampled_img_mode.pt"
model = torch.load(path)

for layer,param in model.state_dict().items():
    print(layer, param)