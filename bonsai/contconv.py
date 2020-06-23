import torch.nn as nn
import torch


class Pruner(nn.Module):
    def __init__(self, m=1e5, mem_size=0, init=None):
        super().__init__()
        if init is None:
            init = .005
        self.mem_size = mem_size
        self.weight = nn.Parameter(torch.tensor([init]))
        self.m = m

        #self.gate = lambda w: torch.relu(w)
        
        self.gate = lambda w: (.5 * w / torch.abs(w)) + .5
        
        self.saw = lambda w: (self.m * w - torch.floor(self.m * w)) / self.m
        self.weight_history = [1]

    def __str__(self):
        return 'Pruner: M={},N={}'.format(self.M, self.channels)

    def num_params(self):
        # return number of differential parameters of input model
        return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.parameters())])

    def track_gates(self):
        self.weight_history.append(self.gate(self.weight).item())

    def get_deadhead(self, verbose=False):
        deadhead = not any(self.weight_history)
        if verbose:
            print(self.weight_history, deadhead)
        self.weight_history = []
        return deadhead

    def deadhead(self):
        for param in self.parameters():
            param.requires_grad = False
            
    def monitor(self):
        return float(self.sg().detach().item())
            

    def sg(self):
        return self.saw(self.weight) + self.gate(self.weight)

    def forward(self, x):
        return self.sg() * x



class ContConv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, groups, stride, padding):
        super().__init__()
        # immutable parameters
        self.dim1 = c_out
        self.dim2 = c_in//groups
        self.groups = groups
        self.padding = padding
        self.stride = stride

        self.unique = False
        
        # mask storage
        self.k = kernel_size
        self.sizes = list(range(1, self.k+1, 2))
        
        self.ks = [str(x) for x in self.sizes]
        self.cs = [str(x) for x in range(self.dim1)]
        
        self.k_masks = {}
        
        if self.unique:
            self.k_pruners = nn.ModuleDict({c: nn.ModuleDict({size: Pruner() for size in self.ks}) for c in self.cs})
            self.d_pruners = nn.ModuleDict({c: Pruner() for c in self.cs})
        else:
            self.k_pruners = nn.ModuleDict({size: Pruner() for size in self.ks})
            self.d_pruners = Pruner()
            
        self.d_mask = {}
        self.gen_masks()

        # adaptive
        self.weight = nn.Parameter(torch.ones([self.dim1, self.dim2, self.k, self.k], requires_grad=True))
        nn.init.kaiming_uniform_(self.weight)
       
     
    def gen_masks(self):
        mid = self.k//2
        for size in self.sizes:
            mask = torch.zeros((self.k,self.k), requires_grad=False).cuda()
            adj = (size-1)//2
            mask[mid-adj:mid+adj+1, mid-adj:mid+adj+1] = 1

            for sub_size in self.sizes[::-1]:
                if sub_size<size:    
                    mask *= (~self.k_masks[str(sub_size)].to(bool)).to(torch.float32)
            self.k_masks[str(size)] = mask
        
        self.d_mask['base'] = torch.zeros((self.k,self.k), requires_grad=False).cuda()
        self.d_mask['base'][::2,::2]=1
        self.d_mask['base'][1:-1:2,1:-1:2]=1
        self.d_mask['dil'] = (~self.d_mask['base'].to(bool)).to(torch.float32)
     
    
    def mask(self):
        if self.unique:
            masks = []
            for c in self.cs:
                d_mask = self.d_mask['base']+self.d_pruners[c](self.d_mask['dil'])
                k_mask = sum([self.k_pruners[c][k](k_mask) for k, k_mask in self.k_masks.items()])
                masks.append(d_mask*k_mask)
            return torch.stack(masks)
        else:
            d_mask = self.d_mask['base']+self.d_pruners(self.d_mask['dil'])
            k_mask = sum([self.k_pruners[k](k_mask) for k, k_mask in self.k_masks.items()])
            return d_mask*k_mask
    
    def forward(self, x):
        return nn.functional.conv2d(
            x,
            self.weight * self.mask(),
            stride=self.stride,
            padding=self.padding,
            groups=self.groups)