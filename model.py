import torch.nn as nn
from torchvision.models import resnet101



class ResNet(nn.Module):
    def __init__(self, img_size, label):
        super(ResNet, self).__init__()
        self.feature_size = img_size //32
        self.label = label
        base_model = resnet101(pretrained=True, progress=False)
        base_model = list(base_model.children())[:-2]
        self.resnet = nn.Sequential(*base_model)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(2048*self.feature_size*self.feature_size, self.label)
        )


    def forward(self, x):
        batch_size = x.size(0)
        x = self.layer1(x)
        x = self.resnet(x)   # output size: B x 2048 x H/32 x W/32
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x