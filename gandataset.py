from torch.utils.data import DataLoader, Dataset
from skimage import io,transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np

class celebA(Dataset):
    def __init__(self,root_dir): #-> None:,transform=None
        #super().__init__()
        self.root_dir=root_dir
        #self.transform=transform
        self.images=os.listdir(self.root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):# -> T_co:
        image_index=self.images[index]
        img_path=os.path.join(self.root_dir,image_index)
        img=io.imread(img_path)
        img=np.array(img)
        return torch.cuda.FloatTensor(img)/255.0
    
    def plot_image(self,index):
        image_index=self.images[index]
        img_path=os.path.join(self.root_dir,image_index)
        img=io.imread(img_path)
        plt.imshow(img,interpolation='nearest')

        pass
    pass

if __name__ == '__main__':
    dataroot='/media/wto/data/datasets/CelebA/Img/img_align_celeba'
    celebdata=celebA(dataroot)
    print(celebdata.__len__())
    celebdata.plot_image(20259)
    plt.show()