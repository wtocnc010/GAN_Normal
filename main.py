
from datetime import time
import torch
from gantest import Discriminator,Generator
import random
import time
import matplotlib.pyplot as plt
import numpy


def generate_real():
    real_data=torch.FloatTensor(
        [random.uniform(0.8,1.0),
        random.uniform(0.0,0.2),
        random.uniform(0.8,1.0),
        random.uniform(0.0,0.2),])
    return real_data

def generate_random(size):
    random_data=torch.rand(size)
    return random_data



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device='cpu'
    print(device)

    grealcuda=generate_real().to(device)
    grandcuda=generate_random(4).to(device) 

    print(grealcuda,grandcuda)

    #鉴别器训练代码  ***
    # D=Discriminator().to(device)
    # for n,p in D.named_parameters():
    #     print(p.device,' ',n)

    # for i in range(10000):

    #     D.train(grealcuda, torch.FloatTensor([1.0]).to(device))

    #     D.train(grandcuda,torch.FloatTensor([0.0]).to(device))
    #     pass

    # D.plot_progress()
    # print( D.forward( grealcuda ).item() )
    # print( D.forward( grandcuda).item() )
    #***
    start=time.time()

    #完整GAN网络，鉴别器+生成器
    D=Discriminator().to(device)
    G=Generator().to(device)
    image_list=[]
    for i in range(10000):
        #用真实样本训练鉴别器
        D.train(grealcuda,torch.FloatTensor([1.0]).to(device))

        #用生成样本训练鉴别器,即强制告诉鉴别器:由生成器生成的都是假的，不管生成器生成的分布多么的逼近GroundTrue.
        D.train(G.forward(torch.FloatTensor([0.5]).to(device)).detach(),torch.FloatTensor([0.0]).to(device))

        #训练生成器,将0.5的初始值输入，经过生成器变成类似样本的分布，再经过鉴别器，输出的结果，和targets做对比。
        G.train(D,torch.FloatTensor([0.5]).to(device),torch.FloatTensor([1.0]).to(device))
        
        #每1000次记录一下生成器的输出
        if (i%1000==0):
            image_list.append(G.forward(torch.FloatTensor([0.5]).to(device)).to('cpu').detach().numpy())
        pass

    end=time.time()

    print('Running time: %s'%(end-start))

    D.plot_progress()
    G.plot_progress()

    print(G.forward(torch.FloatTensor([0.5]).to(device)))

    plt.figure(figsize=(16,8))
    plt.imshow(numpy.array(image_list).T, interpolation='none',cmap='Blues')
    plt.show()