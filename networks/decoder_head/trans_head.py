import torch.nn as nn
import torch
import torch.nn.functional as F
from ipdb import set_trace

# Point_center encode the segmented point cloud
# one more conv layer compared to original paper

class TransHead(nn.Module):
    def __init__(self, in_feat_dim, out_dim=3):
        super(TransHead, self).__init__()
        self.f = in_feat_dim
        self.k = out_dim

        self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()
        return x


def main():
    feature = torch.rand(10, 1896, 1000)
    net = TransHead(in_feat_dim=1896, out_dim=3)
    out = net(feature)
    print(out.shape)

if __name__ == "__main__":
    main()