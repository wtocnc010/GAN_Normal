import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import h5py
import pandas, numpy, random
import matplotlib.pyplot as plt
from gantest import Discriminator,Generator
from gandataset import celebA




dataroot='/media/wto/data/datasets/CelebA/Img/img_align_celeba'
celebdata=celebA(dataroot)

def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data

def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data   

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print("using cuda:", torch.cuda.get_device_name(0))
    
    start=time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #测试鉴别器...
    # D = Discriminator()
    # # move model to cuda device
    # D.to(device)

    # for image_data_tensor in celebdata:
    #     # real data
    #     image_data_tensor=image_data_tensor.view(218*178*3)
    #     D.train(image_data_tensor, torch.cuda.FloatTensor([1.0]))
    #     # fake data
    #     randomtensor=generate_random_image((218,178,3)).view(218*178*3)
    #     D.train(randomtensor, torch.cuda.FloatTensor([0.0]))
    #     pass

    # end=time.time()
    # print('Running time: %s'%(end-start))

    # D.plot_progress()

    # for i in range(4):
    #     image_data_tensor = celebdata[random.randint(0,20000)].view(218*178*3)
    #     print( D.forward( image_data_tensor ).item() )
    #     pass

    # for i in range(4):
    #     randomtensor=generate_random_image((218,178,3)).view(218*178*3)
    #     print( D.forward( randomtensor).item() )
    #     pass
    #...

    #测试生成器的原始像素图...
    # G = Generator()
    # # move model to cuda device
    # G.to(device)

    # output = G.forward(generate_random_seed(100)).view((218,178,3))

    # img = output.detach().cpu().numpy()

    # plt.imshow(img, interpolation='none', cmap='Blues')
    # plt.show()
    #...

    D=Discriminator().to(device)
    G=Generator().to(device)
    image_list=[]
    epochs=3

    for epoch in range(epochs):
        print('epoch=',epoch+1)

        for image_data_tensor in celebdata:
            image_data_tensor=image_data_tensor.view(218*178*3)
            #用真实样本训练鉴别器
            D.train(image_data_tensor,torch.cuda.FloatTensor([1.0]))

            #用生成样本训练鉴别器,即强制告诉鉴别器:由生成器生成的都是假的，不管生成器生成的分布多么的逼近GroundTrue.
            D.train(G.forward(generate_random_seed(100)).to(device).detach(),torch.cuda.FloatTensor([0.0]))

            #训练生成器,将0.5的初始值输入，经过生成器变成类似样本的分布，再经过鉴别器，输出的结果，和targets做对比。
            G.train(D,generate_random_seed(100).to(device),torch.cuda.FloatTensor([1.0]))
            
            # #每1000次记录一下生成器的输出
            # if (i%1000==0):
            #     image_list.append(G.forward(torch.FloatTensor([0.5]).to(device)).to('cpu').detach().numpy())
            pass
        pass

    end=time.time()

    print('Running time: %s'%(end-start))

    D.plot_progress()
    G.plot_progress()


    # plot several outputs from the trained generator
    # plot a 3 column, 2 row array of generated images
    f, axarr = plt.subplots(2,3, figsize=(16,8))
    for i in range(2):
        for j in range(3):
            output = G.forward(generate_random_seed(100)).view((218,178,3))
            img = output.detach().cpu().numpy()
            axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
            pass
        pass
    plt.show()