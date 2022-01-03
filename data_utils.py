import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,Resize,ToTensor
import numpy as np
from PIL import Image

    
def MAEDataLoader(img_dir, global_batch_size=4096,resize_shape = (256,256)):

    """
    VitDataLoader is a generator
    
    params img_dir : string, indicating data folder which holds image data
    params global_batch_size : int , total data being pulled from this generator
        
    """

    #gets files' name
    files = os.listdir(img_dir)

    #transform set up
    transform = Compose([Resize(resize_shape),ToTensor()])

    #iterative parameters setup
    m = len(files)
    idx = 0

    while m > 0:

        img_batch = []

        low_bound = idx * global_batch_size
        up_bound = min((idx+1)*global_batch_size,len(files))

        for i in range(low_bound,up_bound):

            #get image
            img = Image.open(f"{img_dir}/{files[i]}")   #(H,W,C) channel last

            #transform image
            img = transform(img)   #(C,H,W) channel first

            #append image
            img_batch.append(img)

        #update iterative parameters
        m = m - global_batch_size
        idx = idx + 1

        #return (N,C,H,W) : N = global_batch_size dtype = torch.Tensor
        yield torch.stack(img_batch,dim=0) #np.stack(img_batch,axis=0)








if __name__ == "__main__":

    #experimental code
      
    j = 1

    for i in MAEDataLoader(img_dir=f"{os.getcwd()}/data",global_batch_size=128,resize_shape=(224,224)):

        print(f"{j}: {i.shape} ,{type(i)}")

        j = j + 1
        
    


