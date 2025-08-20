import torch

checkpoint = torch.load('utils/efficientnet+meta_03.pth', map_location='cpu')

for k in checkpoint.keys():
    print(k, type(checkpoint[k]))

# checkpoint = torch.load('efficientnet+meta_03.pth', map_location='cpu')
print(checkpoint['tabular_fc.0.weight'].shape)  # ví dụ torch.Size([64, 5])
