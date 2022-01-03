from einops import rearrange,reduce,repeat
from einops.layers.torch import Rearrange,Reduce
from torch import nn
import torch
import numpy as np
from data_utils import *

def get_sinusoid_pos_encoding(n_pos,d_hid):

    """
    params n_pos : H*W e.g. 196,256
    params d_hid : embd size e.g. 768
    """
    angular_vec_k = lambda pos_i : [pos_i/np.power(10000,2*(j//2)/d_hid) for j in range(d_hid)]

    #pos enc set up
    pos_enc_vec = np.linspace(start=0,stop=1,num=n_pos) #(n_pos,)
    pos_enc_vec = [angular_vec_k( pos_enc_vec[i] ) for i in range(n_pos)] #(n_pos,d_hid)
    pos_enc_vec = np.stack(pos_enc_vec,axis=0) #(n_pos,d_hid)

    #apply sinusoid
    pos_enc_vec[:,::2] = np.sin(pos_enc_vec[:,::2]) 
    pos_enc_vec[:,1::2] = np.cos(pos_enc_vec[:,1::2])

    #return torch Tensor (1,n_pos,d_hid)
    return torch.FloatTensor(pos_enc_vec).unsqueeze(0)


class PatchEmbedding(nn.Module):

    def __init__(self,in_channels,out_channels,patch_size=16,img_size=(256,256)):

        """
        params patch_size : int size of patch with both height width equal to patch_size
        params img_size : tuple (int,int) image size in (H,W) format
        """

        super(PatchEmbedding,self).__init__()

        H,W = img_size

        h = H // patch_size
        w = W // patch_size

        self.patch_size = patch_size

        #projection layer (N,C_in,H_in,W_in) -- > (N,M,C_out) M = H_out*W_out
        self.proj_layer = nn.Sequential(nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size=self.patch_size,stride=self.patch_size),
                                        Rearrange("N C H W -> N (H W) C")
                                        )

        #position encoding 
        self.pos_enc_tensor = get_sinusoid_pos_encoding(h*w,out_channels) #(1,H*W,C)
    
    def forward(self,x):

        """
        params x : input tensor (N,C,H,W)
        """
        N,_,_,_ = x.shape

        # projection : (N,C_in,H_in,W_in) -- > (N,M,C_out) M = H_out*W_out
        x = self.proj_layer(x)

        #position encoding (1,H*W,C) --> (N,H*W,C)
        pos_enc_expand = self.pos_enc_tensor.expand(N,-1,-1).type_as(x).to(x.device).clone().detach()

        x = x + pos_enc_expand

        return x




if __name__ == "__main__":

    #experimental code
    angular_vec_k = lambda pos_i : [pos_i/np.power(10000,2*(j//2)/768) for j in range(768)]

    print(len(angular_vec_k(1)))

    t = get_sinusoid_pos_encoding(224,768)

    data = torch.randn(16,3,224,224)
    t = PatchEmbedding(3,768,img_size=(224,224))

    a = t(data)

    
