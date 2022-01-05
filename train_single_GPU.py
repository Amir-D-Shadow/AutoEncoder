from modules import *
from data_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import time
import os

class Encoder(nn.Module):

    def __init__(self,in_patch_emb,out_patch_emb,in_dim,out_dim,num_ViTblk=12,emb_size=768,num_heads=12,drop_rate=0.5,patch_size=16,img_size=(256,256)):

        super(Encoder,self).__init__()

        #patch embedding parameter
        self.in_patch_emb = in_patch_emb
        self.out_patch_emb = out_patch_emb
        self.patch_size = patch_size
        self.img_size = img_size

        
        #Block's parameter
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        self.num_ViTblk = num_ViTblk

        #set up patch embeding editor
        self.patch_emb_layer = PatchEmbedding(in_channels = self.in_patch_emb,
                                              out_channels = self.out_patch_emb,
                                              patch_size = self.patch_size,
                                              img_size = self.img_size
                                              )
        

        #set up encoder
        self.blocks = []

        for i in range(self.num_ViTblk):

            self.blocks.append( Block(in_dim = self.in_dim,
                                      out_dim = self.out_dim,
                                      emb_size = self.emb_size,
                                      num_heads = self.num_heads,
                                      drop_rate = self.drop_rate
                                      )
                                )

        self.blocks = nn.ModuleList(self.blocks)
        

    def forward(self,x):

        """
        x : Tensor (N,C,H,W) , e.g. masked image
        """

        #patch embedding
        enc_out = self.patch_emb_layer(x) # (N,h*w,C), C = emb_size

        #encoding (ViT transformer encoder blocks)
        for i in range(self.num_ViTblk):

            enc_out = self.blocks[i](enc_out) #(N,M,H*D) , H*D = emb_size , M = num of patches

        #Tensor (N,M,H*D) , H*D = emb_size , M = num of patches
        return enc_out



class BottleNeck(nn.Module):

    def __init__(self,in_dim,emb_size):

        super(BottleNeck,self).__init__()

        self.emb_size = emb_size
        self.in_dim = in_dim

        self.enc_to_dec = nn.Sequential(nn.Linear(in_features = self.in_dim,out_features = self.emb_size),
                                        nn.LayerNorm(self.emb_size),
                                        nn.GELU()
                                        )

    def forward(self,x,masked_token):

        """
        x : Tensor, encoder output (N,m1,H*D)
        masked_token : Tensor , masked patches (N,m2,C) C = patch_size * patch_size * 3
        """

        N,m1,C = x.shape
        _,m2,_ = masked_token.shape

        #total num of patches
        M = m1 + m2

        #get full set of image 
        x_full = torch.cat([x,masked_token],dim=1)

        #position encoding
        pos_enc_tensor = get_sinusoid_pos_encoding(M,C)
        pos_enc_tensor = pos_enc_tensor.expand(N,-1,-1).type_as(x).to(x.device).clone().detach()

        x_full = x_full + pos_enc_tensor

        #embedding (N,M,emb_size)
        x_out = self.enc_to_dec(x_full)

        return x_out

        
    
class Decoder(nn.Module):

    def __init__(self,in_dim,out_dim,final_out,num_ViTblk=4,emb_size=768,num_heads=12,drop_rate=0.5):

        """
        params final-out : int , final linear layer output dim
        """

        super(Decoder,self).__init__()

        
        #Block's parameter
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        self.num_ViTblk = num_ViTblk
        self.final_out = final_out

        #set up Decoder
        self.blocks = []

        for i in range(self.num_ViTblk):

            self.blocks.append( Block(in_dim = self.in_dim,
                                      out_dim = self.out_dim,
                                      emb_size = self.emb_size,
                                      num_heads = self.num_heads,
                                      drop_rate = self.drop_rate
                                      )
                                )

        self.blocks = nn.ModuleList(self.blocks)
        
        #output layer
        self.out_layer = nn.Sequential(nn.Linear(in_features = self.out_dim,out_features = self.final_out),
                                       nn.LayerNorm(self.final_out),
                                       nn.GELU()
                                       )


    def forward(self,x):

        """
        x : Tensor (N,M,emb_size) , e.g. encoder to decoder output
        """

        dec_out = x
        
        #decoding
        for i in range(self.num_ViTblk):

            dec_out = self.blocks[i](dec_out) #(N,M,emb_size), M = num of patches

        #output
        dec_out = self.out_layer(dec_out)
        
        #Tensor (N,M,emb_size), M = num of patches
        return dec_out
    

def mse2psnr(mse):
    """
    :param mse: scalar
    :return:    scalar np.float32
    """
    mse = np.maximum(mse, 1e-10)  # avoid -inf or nan when mse is very small.
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)


def train_one_epoch(enc_model,bn_model,dec_model,opt_enc,opt_bn,opt_dec,data_dir,emb_size=768,global_batch_size=256,pos_enc=True,resize_shape = (256,256)):

    enc_model.train()
    bn_model.train()
    dec_model.train()

    total_loss = []
    
    for input_img,gt_image in MAEDataLoader(img_dir=data_dir,global_batch_size=global_batch_size,pos_enc=pos_enc,resize_shape=resize_shape):

        #set input info: input image size
        N,C,H,W = input_img.shape
        input_info = (N,C,H,W)
        
        #keeping 25% , masking 75%
        h = H//2
        w = W//2

        n_vis = h*w

        #get mask : bool (N,C,H,W)
        vis = RandomMasking(n_vis = n_vis,input_info = input_info)
        
        mask = 1 - vis
        mask = mask.astype(np.bool_)

        if pos_enc :

            mask = mask[:,:-1,:,:]

        vis = torch.from_numpy(vis)
        mask = torch.from_numpy(mask)

        #mask img
        input_img_vis = input_img[vis].view(N,C,h,w) #(N,C,h,w)
        input_img_vis = input_img_vis.to(device)

        #masked token
        if pos_enc:
            
            input_img_masked = input_img[:,:-1,:,:][mask].view(N,-1,emb_size) #(N,M,emb_size)

        else:

            input_img_masked = input_img[mask].view(N,-1,emb_size) #(N,M,emb_size)

        input_img_masked = torch.from_numpy( np.zeros_like( input_img_masked.numpy() ) )
        input_img_masked = input_img_masked.to(device)

        #training
        step1 = enc_model(input_img_vis) # out: (N,m1,C)
        step2 = bn_model(step1,input_img_masked) # out: (N,M,C)
        step3 = dec_model(step2) #out: (N,M,C)

        img_pred = step3.view(N,3,H,W)
        gt_image = gt_image.to(device)

        #calculate loss
        L2_loss = F.mse_loss(img_pred,gt_image)

        #optimize
        L2_loss.backward()
        
        opt_dec.step()
        opt_bn.step()
        opt_enc.step()

        opt_dec.zero_grad()
        opt_bn.zero_grad()
        opt_enc.zero_grad()

        #collect step loss
        total_loss.append(L2_loss)

    total_loss = torch.stack(total_loss).mean().item()
    
    return total_loss


if __name__ == "__main__":

    #experimental code    
    """
    data = torch.randn(64,4,128,128).to("cuda:0")
    masked_token = torch.randn(64,128,768)

    enc1 = Encoder(in_patch_emb=4,out_patch_emb=768,in_dim=768,out_dim=768,num_ViTblk=12,emb_size=768,num_heads=12,drop_rate=0.5,patch_size=16,img_size=(128,128)).to(data.device)
    bn1 = BottleNeck(in_dim=768,emb_size=768).to(data.device)
    dec1 = Decoder(in_dim = 768 ,out_dim=768,final_out=768,num_ViTblk=4,emb_size=768,num_heads=12,drop_rate=0.5).to(data.device)

    step1 = enc1(data)
    step2 = bn1(step1,masked_token)
    step3 = dec1(step2)
    """

    #training

    #set up
    total_epochs = 5
    global_batch = 64
    #batch_size = 64

    device = "cuda:7"

    data_dir = f"{os.getcwd()}/data"
    save_weight_path = f"{os.getcwd()}/model_weights"

    #initialize encoder
    enc_model = Encoder(in_patch_emb=4,
                        out_patch_emb=768,
                        in_dim=768,
                        out_dim=768,
                        num_ViTblk=12,
                        emb_size=768,
                        num_heads=12,
                        drop_rate=0.5,
                        patch_size=16,
                        img_size=(128,128)
                        )

    enc_model.to(device)
    
    #initialize bottle neck
    bn_model = BottleNeck(in_dim=768,
                          emb_size=768
                          )

    bn_model.to(device)

    #initialize decoder
    dec_model = Decoder(in_dim = 768 ,
                        out_dim=768,
                        final_out=768,
                        num_ViTblk=4,
                        emb_size=768,
                        num_heads=12,
                        drop_rate=0.5
                        )

    dec_model.to(device)

    #set optimier
    opt_enc = torch.optim.Adam(enc_model.parameters(),lr = 0.001)
    opt_bn = torch.optim.Adam(bn_model.parameters(),lr = 0.001)
    opt_dec = torch.optim.Adam(dec_model.parameters(),lr = 0.001)

    #set lr scheduler
    scheduler_enc = torch.optim.lr_scheduler.MultiStepLR(opt_enc,milestones=list(range(0, 10000, 10)), gamma=0.9)
    scheduler_bn = torch.optim.lr_scheduler.MultiStepLR(opt_bn,milestones=list(range(0, 10000, 10)), gamma=0.9)
    scheduler_dec = torch.optim.lr_scheduler.MultiStepLR(opt_dec,milestones=list(range(0, 10000, 10)), gamma=0.9)

    print(f"Start Training {total_epochs} epochs")
    for epoch_i in range(total_epochs):

        start_time = time.time()

        total_loss = train_one_epoch(enc_model = enc_model,
                                     bn_model = bn_model,
                                     dec_model = dec_model,
                                     opt_enc = opt_enc,
                                     opt_bn = opt_bn,
                                     opt_dec = opt_dec,
                                     data_dir = data_dir,
                                     emb_size=768,
                                     global_batch_size=global_batch,
                                     pos_enc=True,
                                     resize_shape = (256,256),
                                     input_info = (256,4,256,256))

        scheduler_dec.step()
        scheduler_bn.step()
        scheduler_enc.step()

        #saving checkpoints
        torch.save(enc_model.state_dict(),f"{save_weight_path}/Encoder.pt")
        torch.save(bn_model.state_dict(),f"{save_weight_path}/BottleNet.pt")
        torch.save(dec_model.state_dict(),f"{save_weight_path}/Decoder.pt")

        #print info
        print("Epoch {epoch_i} : Training MSE : {total_loss}, estimated time : {time.time() - start_time}")

    print("Training Completed!!")
