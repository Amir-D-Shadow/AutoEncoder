import os
import shutil

path = f"{os.getcwd()}/data"
save = f"{os.getcwd()}/img"

idx = 1

for i in os.listdir(path):

    for j in os.listdir(f"{path}/{i}"):

        for k in os.listdir(f"{path}/{i}/{j}"):

            for img_name in os.listdir(f"{path}/{i}/{j}/{k}"):

                shutil.copyfile(f"{path}/{i}/{j}/{k}/{img_name}",f"{save}/img_{idx}.jpg")

                print(idx)
                idx = idx + 1

        

    
print(idx)
