import torch.nn as nn
import torch
import torch.nn.functional as F
from ipdb import set_trace


class RotHead(nn.Module):
    def __init__(self, in_feat_dim, out_dim=3):
        super(RotHead, self).__init__()
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

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()

        return x


def main():
    points = torch.rand(2, 1350, 1024)  # batchsize x feature x numofpoint
    rot_head = RotHead(in_feat_dim=1350, out_dim=3)
    rot = rot_head(points)
    print(rot.shape)


if __name__ == "__main__":
    main()