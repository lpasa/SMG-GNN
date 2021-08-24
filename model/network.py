from torch import nn
import torch

class SMGNetwork(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_classes,
                 dropout,
                 k,
                 convLayer,
                 out_fun = nn.Softmax(dim=1),
                 device=None,
                 norm=None,
                 bias=False):
        super(SMGNetwork, self).__init__()
        self.g=g
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device=device
        self.layers = nn.ModuleList()
        self.out_fun=out_fun
        self.dropout = nn.Dropout(p=dropout)
        self.layer= convLayer(in_feats, n_classes, k, cached=True, bias=bias, norm=norm)

    def forward(self,features):
        h = features
        h=self.dropout(h)
        h,layer_entropy = self.layer(self.g,h)
        softmax_entropy=layer_entropy
        return self.out_fun(h),h, torch.mean(softmax_entropy)
