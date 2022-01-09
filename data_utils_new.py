import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,Resize,ToTensor
import numpy as np
from PIL import Image
import random

def get_sinusoid_pos_encoding_for_img(n_pos):

    """
    params n_pos : H*W e.g. 224*224,256*256
    """
    angular_vec_k = lambda pos_i : pos_i/np.power(10000,2*(pos_i//2)/n_pos) 

    #pos enc set up
    pos_enc_vec = np.linspace(start=0,stop=1,num=n_pos) #(n_pos,)
    pos_enc_vec = [angular_vec_k( pos_enc_vec[i] ) for i in range(n_pos)] #(n_pos,)
    pos_enc_vec = np.array(pos_enc_vec) #(n_pos,)

    #apply sinusoid
    pos_enc_vec[::2] = np.sin(pos_enc_vec[::2]) 
    pos_enc_vec[1::2] = np.cos(pos_enc_vec[1::2])

    #return numpy array (n_pos,)
    return pos_enc_vec


def MAEDataLoader(img_dir, global_batch_size=4096,pos_enc=False,resize_shape = (256,256)):

    """
    VitDataLoader is a generator
    
    params img_dir : string, indicating data folder which holds image data
    params global_batch_size : int , total data being pulled from this generator
        
    """

    H,W = resize_shape

    #transform set up
    transform = Compose([Resize(resize_shape),ToTensor()])

    if pos_enc:

        pos_enc_t = get_sinusoid_pos_encoding_for_img(H*W).reshape(1,H,W)  #(1,H,W)
        pos_enc_t = torch.from_numpy(pos_enc_t)


    count = 0
    input_batch = []
    gt_batch = []
    
    for i in os.listdir(img_dir):

        for j in os.listdir(f"{img_dir}/{i}"):

            for k in os.listdir(f"{img_dir}/{i}/{j}"):

                for img_name in os.listdir(f"{img_dir}/{i}/{j}/{k}"):

                    #get image
                    img = Image.open(f"{img_dir}/{i}/{j}/{k}/{img_name}")   #(H,W,C) channel last

                    #transform image
                    img = transform(img)   #(C,H,W) channel first

                    #append img for ground truth
                    gt_batch.append(img)

                    #pos enc
                    if pos_enc:

                        img = torch.cat([img,pos_enc_t],dim=0) #(C+1,H,W)

                    #append image for input
                    input_batch.append(img)
                    
                    count = count + 1

                    if count == global_batch_size:
                        

                        yield (torch.stack(input_batch,dim=0).type(torch.float32), torch.stack(gt_batch,dim=0).type(torch.float32) )

                        input_batch = []
                        gt_batch = []

                        count = 0

    #return input_batch , gt_batch  (N,C*,H,W) : N = global_batch_size dtype = torch.Tensor
    yield (torch.stack(input_batch,dim=0).type(torch.float32), torch.stack(gt_batch,dim=0).type(torch.float32) )#np.stack(input_batch,axis=0)



def RandomMasking(n_vis,input_info = (16,3,256,256)):

    """
    params n_vis : int ,  number of visible pixels
    params input_info : tuple (int,int,int,int) , (N,C,H,W) <-channel last form
    """
    N,C,H,W = input_info

    M_pixels = int(H*W)
    
    n_masked = M_pixels - n_vis

    FinalMask = []

    for i in range(N):

        mask = np.hstack([np.zeros(n_masked),np.ones(n_vis)]) #(H*W,)

        np.random.shuffle(mask)

        mask = mask.reshape(H,W) #(H,W)

        mask_list = [mask for i in range(C)]
        
        mask_list = np.stack(mask_list,axis=0) #(C,H,W)

        #test code
        #print(f"val comp: {np.allclose(mask_list[1,:,:],mask)} , {mask_list.shape}")

        #save mask
        FinalMask.append(mask_list)

    FinalMask = np.stack(FinalMask,axis=0).astype(np.bool_) #(N,C,H,W)

    #numpy array (N,C,H,W) bool type
    return FinalMask


if __name__ == "__main__":

    #experimental code
      
    j = 1

    for i,k in MAEDataLoader(img_dir=f"{os.getcwd()}/data",global_batch_size=1024,pos_enc=True,resize_shape=(256,256)):

        print(f"{j}: val_com:{np.allclose(i.numpy()[:,:-1,:,:],k.numpy())} input:{i.shape} ,{type(i)}, gt:{k.shape},{type(k)}")

        j = j + 1
        
    
    t = get_sinusoid_pos_encoding_for_img(256*256).reshape(1,256,256)

    result = RandomMasking(n_vis = 16384,input_info=(8,4,256,256))
