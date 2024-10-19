from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch import nn 
import torch 
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

class Net(nn.Module):   # 模型定义
    def __init__(self):
        super(Net,self).__init__() 
        self.flatten=nn.Flatten()
        self.seq=nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )
    
    def forward(self,x):
        x=self.flatten(x)
        return self.seq(x)

def main():
    dist.init_process_group(backend='gloo') # 【集合通讯】其他进程连master，大家互认
    
    rank=dist.get_rank()
    world_size=dist.get_world_size()
    
    checkpoint=None # 各自加载checkpoint
    try:
        checkpoint=torch.load('checkpoint.pth')   # mapping location for CPU
    except:
        pass
    
    model=Net()
    if checkpoint and rank==0:  # rank0恢复模型参数
        model.load_state_dict(checkpoint['model'])

    model=DDP(model) # 【集合通讯】rank0广播参数给其他进程
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001) #model参数一致，则optim会保证其初始状态一致
    if checkpoint: 
        optimizer.load_state_dict(checkpoint['optimizer'])  # 各自加载checkpoint

    train_dataset=MNIST(root='./data',download=True,transform=ToTensor(),train=True) # 各自加载dataset
    sampler=DistributedSampler(train_dataset) # 指派子集给各进程
    train_dataloader=DataLoader(train_dataset,batch_size=32,sampler=sampler,persistent_workers=True,num_workers=2)
    
    val_dataset=MNIST(root='./data',download=True,transform=ToTensor(),train=False)
    val_dataloader=DataLoader(val_dataset,batch_size=32,shuffle=True,persistent_workers=True,num_workers=2)

    for epoch in range(20):
        sampler.set_epoch(epoch)    # 【集合通讯】生成随机种子，rank0广播给其他进程
        
        model.train()
        for x,y in train_dataloader:
            pred_y=model(x) # 【集合通讯】rank0广播model buffer给其他进程
            loss=F.cross_entropy(pred_y,y)
            optimizer.zero_grad()
            loss.backward() # 【集合通讯】每个参数的梯度做all reduce（每个进程会收到其他进程的梯度，并求平均）
            optimizer.step()
        
        dist.reduce(loss,dst=0) # 【集合通讯】rank0汇总其他进程的loss
        
        if rank==0: 
            train_avg_loss=loss.item()/world_size
            
            # evaluate
            raw_model=model.module
            val_loss=0
            with torch.no_grad():
                for x,y in val_dataloader:
                    pred_y=raw_model(x)
                    loss=F.cross_entropy(pred_y,y)
                    val_loss+=loss.item()
            val_avg_loss=val_loss/len(val_dataloader)
            print(f'train_loss:{train_avg_loss} val_loss:{val_avg_loss}')
            
            # checkpoint
            torch.save({'model':model.module.state_dict(),'optimizer':optimizer.state_dict()},'.checkpoint.pth')
            os.replace('.checkpoint.pth','checkpoint.pth')
        
        dist.barrier() # 【集合通讯】等待rank0跑完eval
        
# torchrun --nproc-per-node 8 singlenode.py
if __name__=='__main__':
    main()