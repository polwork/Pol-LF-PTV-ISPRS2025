import torch

class Convection_backbone(torch.nn.Module):
    def __init__(self, dim_x=3, filter_size=128, act_fn='relu', layer_size=8):
        super().__init__()
        self.layer_size = layer_size
        
        act_fn = 'PReLU'
        if act_fn == 'relu':
            act_fn = torch.nn.LeakyReLU
        elif act_fn == 'sigmoid':
            act_fn = torch.nn.Sigmoid
        elif act_fn == 'PReLU':
            act_fn = torch.nn.PReLU
        elif act_fn == 'GLU':
            act_fn = torch.nn.GLU
        
        self.nn_layers = torch.nn.ModuleList([])
        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size)))
            self.nn_layers.append(act_fn())
            for _ in range(layer_size-1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size)))
                self.nn_layers.append(act_fn())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x)))

    def forward(self, x):
        """ points -> features
            [B, N, 3] -> [B, K]
        """
        for layer in self.nn_layers:
            x = layer(x)


        return x
    