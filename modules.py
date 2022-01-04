from einops import rearrange,reduce,repeat
from einops.layers.torch import Rearrange,Reduce
from torch import nn
import torch
import numpy as np
import os

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

        #projection layer (N,C_in,H_in,W_in) -- > (N,M,C_out) M = h*w , C_out = emb_size
        self.proj_layer = nn.Sequential(nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size=self.patch_size,stride=self.patch_size),
                                        nn.LeakyReLU(),
                                        Rearrange("n c h w -> n (h w) c")
                                        )

        #position encoding 
        self.pos_enc_tensor = get_sinusoid_pos_encoding(h*w,out_channels) #(1,h*w,C)
    
    def forward(self,x):

        """
        params x : input tensor (N,C,H,W)
        """
        N,_,_,_ = x.shape

        # projection : (N,C_in,H_in,W_in) -- > (N,M,C_out) M = h*w = num of patches
        x = self.proj_layer(x)

        #position encoding (1,h*w,C) --> (N,h*w,C)  C = emb_size
        pos_enc_expand = self.pos_enc_tensor.expand(N,-1,-1).type_as(x).to(x.device).clone().detach()

        x = x + pos_enc_expand

        return x # (N,h*w,C), C = emb_size


class MultiHeadSelfAttention(nn.Module):

    def __init__(self,in_dim,emb_size=768,num_heads=12,drop_rate=0.5):

        """
        params in_dim : int , input dimension
        params emb_size : int , embedding size
        params heads : int , num of heads for attention
        params drop_rate : float , probability to be dropped in dropout layer
        """

        super(MultiHeadSelfAttention,self).__init__()

        self.emb_size = emb_size
        self.num_of_heads = num_heads
        
        self.qkv_encoding = nn.Linear(in_features = in_dim,out_features = emb_size*3)
        self.softmax_att = nn.Softmax(dim=-1)
        self.att_drop = nn.Dropout(p=drop_rate)
        self.att_resize = Rearrange("n h m d -> n m (h d)")
        self.proj_layer = nn.Linear(in_features = emb_size,out_features = emb_size)

    def forward(self,x):

        """
        x : Tensor , (N,M,C) ,  M = h*w (number of patches)
        """

        qkv = self.qkv_encoding(x) # (N,M,3*H*D)

        #o : qkv (num of operational tensor), n : batch size , m : num of patches , d : hidden dims
        qkv = rearrange(qkv,"n m (h d o) -> (o) n h m d",h=self.num_of_heads,o=3) # (3,N,H,M,D)

        #Notice that H*D = emb_size
        query = qkv[0] # (N,H,M,D) <==> (N,H,Q,D)
        key = qkv[1]   # (N,H,M,D) <==> (N,H,K,D)
        value = qkv[2] # (N,H,M,D) <==> (N,H,V,D)

        #scaling 
        scale = (self.emb_size)**(0.5)
        query = query / scale

        #attention process 
        att_weight = torch.einsum("nhqd,nhkd->nhqk",query,key) #(N,H,Q,K)
        att_weight = self.softmax_att(att_weight) #(N,H,Q,K)
        att_weight = self.att_drop(att_weight)    #(N,H,Q,K)

        att = torch.einsum("nhqv,nhvd->nhqd",att_weight,value) # (N,H,M,D) <==> (N,H,Q,D)
        att = self.att_resize(att) # (N,M,H*D)
        att = self.proj_layer(att) # (N,M,H*D)

        #return tensor (N,M,H*D) , H*D = emb_size
        return att
        

class MLP(nn.Module):

    def __init__(self,in_dim,hidden_dim,out_dim,drop_rate=0.5):

        """
        params expansion : int , usually scaled input dim produce hidden dim
        """

        super(MLP,self).__init__()
        
        self.fc1 = nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.act_fn = nn.GELU()
        self.drop1 = nn.Dropout(p=drop_rate)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self,x):

        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.drop1(x)
        x = self.fc2(x)

        return x


class Block(nn.Module):

    def __init__(self,in_dim,out_dim,emb_size=768,num_heads=12,drop_rate=0.5):

        super(Block,self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.drop_rate = drop_rate

        self.norm_att = nn.Sequential(nn.LayerNorm(self.emb_size),
                                      MultiHeadSelfAttention(in_dim=self.in_dim,emb_size=self.emb_size,num_heads=self.num_heads,drop_rate=self.drop_rate),
                                      nn.Dropout(p=self.drop_rate)
                                      )
        
        self.norm_feedforward = nn.Sequential(nn.LayerNorm(self.emb_size),
                                              MLP(in_dim=self.emb_size,hidden_dim=int(self.emb_size*4),out_dim=self.out_dim,drop_rate=self.drop_rate),
                                              nn.Dropout(p=self.drop_rate)
                                              )
        
    def forward(self,x):

        """
        x : Tensor , embedded input patches with shape (N,h*w,C) , C = emb_size
        """

        res = x #(N,h*w,C)

        carrier_1 = self.norm_att(x) #(N,M,H*D)
        carrier_1 = carrier_1 + res  #(N,M,H*D)

        res = carrier_1

        carrier_2 = self.norm_feedforward(carrier_1) #(N,M,H*D)
        carrier_2 = carrier_2 + res                  #(N,M,H*D)

        return carrier_2


    
        
if __name__ == "__main__":

    #experimental code
    angular_vec_k = lambda pos_i : [pos_i/np.power(10000,2*(j//2)/768) for j in range(768)]

    print(len(angular_vec_k(1)))

    t = get_sinusoid_pos_encoding(256*256,1)#get_sinusoid_pos_encoding(224,768)

    data = torch.randn(16,3,224,224)
    
    step1 = PatchEmbedding(3,768,img_size=(224,224))(data)
    step2 = MultiHeadSelfAttention(in_dim=768,emb_size=768,num_heads=12)(step1)

    step3 = Block(in_dim=768,out_dim=768,emb_size=768,num_heads=12)(step1)
    
