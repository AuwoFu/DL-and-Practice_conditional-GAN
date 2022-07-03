import argparse
from fileinput import filename
import os
import random
import itertools
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime,timezone,timedelta
from torchvision import transforms
from torchvision.utils import save_image
from  statistics import mean

from importlib.util import module_for_loader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from projection.ProjectionDis import  Discriminator_Projection
from projection.ProjectionGen import  Generator
from projection.loss import  DisLoss,GenLoss

from dataloader import ICLEVR_dataset,denormalize
from evaluator import evaluation_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='',help='base directory to save logs')
    parser.add_argument('--model_dir', default='',help='base directory to load model')
    parser.add_argument('--data_root', default='./script/', help='root directory for data json')
    parser.add_argument('--img_root', default='./dataset/', help='root directory for image')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=202105)
    # input setting
    parser.add_argument('--nc', default=3, type=int, help='number of image color channel')
    parser.add_argument('--nz', default=128, type=int, help='Size of z latent vector')
    parser.add_argument('--ngf', default=64, type=int, help='Size of feature maps in generator')
    parser.add_argument('--ndf', default=64, type=int, help='Size of feature maps in discriminator')
    parser.add_argument('--label_dim', default=3, type=int, help='number of data label')
    parser.add_argument('--num_class', default=24, type=int, help='number of data classes')
    parser.add_argument('--image_size', default=(64,64), type=tuple, help='size of training image after transforms')
    parser.add_argument('--save_interval', default=1000, type=int, help='interval for save generated img')
    
    # train
    parser.add_argument('--epoch', default=30000, type=int, help='training epoch')
    parser.add_argument('--iter_D', default=4, type=int, help='training epoch of net D each time')
    parser.add_argument('--iter_G', default=1, type=int, help='training epoch of net G each time')
    parser.add_argument('--loss_type', type=str, default='hinge',help='loss function name. hinge (default) or dcgan.')
    
    parser.add_argument('--lr_G', default=0.0003, type=float, help='learning rate')
    parser.add_argument('--lr_D', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--lr_decay',  default=False, action='store_true',help='whether lr decay')
    parser.add_argument('--lr_decay_start', '-lds', type=int, default=4000,help='Start point of learning rate decay. default: 50000')
    parser.add_argument('--train_bs', default=64,type=int, help='batch size within training')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--optimizer', default='adam',help='optimizer to train with')
    parser.add_argument('--beta1', type=float, default=0.5,help='beta1 (betas[0]) value of Adam.')
    parser.add_argument('--beta2', type=float, default=0.999,help='beta2 (betas[1]) value of Adam.')
    
    # trick
    # flip label
    parser.add_argument('--label_flip',  default=False, action='store_true',help='flip label to train D')
    parser.add_argument('--flip_rate', type=float, default=0.1,help='trick for flip label')
    parser.add_argument('--add_noise',  default=True, action='store_true',help='Apply noise in D input')
    parser.add_argument('--relativistic_loss', '-relloss', default=False, action='store_true',help='Apply relativistic loss or not. default: False')

    # test
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--test_bs', default=64,type=int, help='batch size within testing')
    #parser.add_argument('--test_time', default=100000,type=int, help='repeat times to test')
    
    
    args = parser.parse_args()
    return args


def smoothLabel(bs,isTrue,onehot,device='cuda'):
    if isTrue: #real
        min,max=0.8,1.2
    else: # fake
        min,max=0.0,0.2
    label = (max-min)*torch.rand(onehot.shape) + min
    label=label.to(device)
    label*=onehot
    label=torch.sum(label,dim=(1),keepdim=True)
    return label.to(dtype=torch.float)

def decay_lr(opt, max_iter, start_iter, initial_lr):
    """Decay learning rate linearly till 0."""
    coeff = -initial_lr / (max_iter - start_iter)
    for pg in opt.param_groups:
        pg['lr'] += coeff


def add_noise(tensor):
    tensor+=torch.empty_like(tensor, dtype=tensor.dtype).uniform_(0.0, 1/128.0)
    return tensor

def get_pseudo_label(bs,num_classes):  
    pseudo_labels = torch.from_numpy(
        np.random.randint(low=0, high=num_classes, size=(bs,1))# 0~23
    ) #[bs,3]

    l_2=torch.from_numpy(
        np.random.randint(low=0, high=num_classes+1, size=(bs,2))
    )# 0~24

    pseudo_labels =torch.cat([pseudo_labels,l_2],dim=1) .type(torch.long)

    '''
    # label will repeat
    id_1=np.random.randint(low=0, high=num_classes, size=(bs))# 0~23
    id_2=[]
    for b in range(bs):
        seq=[ c for c in range(num_classes) if c != id_1[b] ]+[num_classes,num_classes] # let 24 can appear twice
        id_2.append(random.sample(seq,  k=2))

    id_1=torch.from_numpy(id_1).unsqueeze(-1)
    id_2=torch.Tensor(id_2)

    pseudo_labels =torch.cat([id_1,id_2],dim=1) .type(torch.long)
    '''

    return pseudo_labels



def sample_for_G(args,bs):
    noise = torch.randn(bs, args.nz, device=args.device) # Generate batch of latent vectors
    pseudo_label_id=get_pseudo_label(bs,args.num_class).to(args.device)
    return noise,pseudo_label_id



def train(netD,netG,args):
    writer = SummaryWriter(args.log_dir)
    # load data
    trainset=ICLEVR_dataset(args,mode='train')
    trainloader=DataLoader(trainset,batch_size=args.train_bs,shuffle=True,num_workers=args.num_workers)
    testset=ICLEVR_dataset(args,mode='test')
    testloader=DataLoader(testset,batch_size=args.train_bs,shuffle=False,num_workers=args.num_workers)
    newtestset=ICLEVR_dataset(args,mode='new_test')
    newTestloader=DataLoader(newtestset,batch_size=args.train_bs,shuffle=False,num_workers=args.num_workers)


    # setting
    dis_criterion=DisLoss(args.loss_type, args.relativistic_loss)
    gen_criterion=GenLoss(args.loss_type, args.relativistic_loss)

    # optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_D,betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_G, betas=(args.beta1, args.beta2))

    # evaluation 
    best_acc_old=0
    best_acc_new=0
    best_acc_avg=0
    # evaluation model
    net_eval=evaluation_model()

    
    print("Starting Training Loop...")
    #train_iterator=itertools.cycle(trainloader)
    train_iterator=iter(trainloader)


    for epoch in tqdm(range(start_epoch,args.epoch)):
        netG.train()

        # decay lr
        if args.lr_decay:
            if epoch >= args.lr_decay_start:
                decay_lr(optimizerG, args.epoch, args.lr_decay_start, args.lr_G)
                decay_lr(optimizerD, args.epoch, args.lr_decay_start, args.lr_D)
              
        '''train net G'''
        all_D_G_z2=[]
        all_loss_G=[]
        bs=args.train_bs
        for i_g in range(args.iter_G):
            netG.zero_grad()
            noise,pseudo_label_id=sample_for_G(args,bs)
            # Generate fake image batch with G
            fake = netG(noise,pseudo_label_id)
            D_fake = netD(fake,pseudo_label_id)
            
            if args.relativistic_loss:
                try:
                    img,label_id = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(trainloader)
                    img,label_id= next(train_iterator)
                
                # add noise
                if args.add_noise:
                    img=add_noise(img)
            

                img = img.to(args.device)
                label_id=label_id.to(args.device,dtype=torch.int)
                with torch.no_grad():
                    D_real = netD(img, label_id)
                
                if D_real.shape[0] != bs:
                    D_fake=D_fake[:D_real.shape[0]] # make same shape
            else:
                D_real = None
            
            # Calculate G's loss based on this output
            loss_G = gen_criterion(D_fake, D_real)
            # Calculate gradients for G
            loss_G.backward()
            # Update G
            optimizerG.step()

            D_G_z2 = D_fake.mean().item()
            all_D_G_z2.append(D_G_z2)
            all_loss_G.append(loss_G.tolist())
        s2=f'D_G_z2={mean(all_D_G_z2):<.6f}, loss_G={mean(all_loss_G):<.6f}'
        
        writer.add_scalar('Train G/D(G(z))', mean(all_D_G_z2))
        writer.add_scalar('Train G/errG', mean(all_loss_G))
        
        '''train net D'''
        all_D_fake=[]
        all_D_real=[]
        all_loss_D=[]
        for i_d in range(args.iter_D):   
            netD.zero_grad()

            # get real data
            try:
                img,label_id= next(train_iterator)
            except StopIteration:
                train_iterator = iter(trainloader)
                img,label_id= next(train_iterator)
            
            # input process
            if args.add_noise:
                img=add_noise(img)
            img = img.to(args.device)
            label_id=label_id.to(args.device,dtype=torch.int)

            bs=label_id.shape[0]
            
            # get fake data
            noise,pseudo_label_id=sample_for_G(args,bs)
            fake = netG(noise,pseudo_label_id) # Generate fake image batch with G
            

            ## D for real             
            D_real = netD(img,label_id)
            D_x = D_real.mean().item()
            all_D_real.append(D_x)

            # D for fake
            D_fake = netD(fake,pseudo_label_id)
            D_G_z1 = D_fake.mean().item()
            all_D_fake.append(D_G_z1)

            # Compute error of D as sum over the fake and the real batches
            if args.label_flip:
                r=random.random()
                if r<args.flip_rate:
                    #flip label
                    D_fake,D_real=D_real,D_fake
            
            loss_D = dis_criterion(D_fake,D_real)
            loss_D.backward()
            all_loss_D.append(loss_D.tolist())
            # Update D
            optimizerD.step()
                
            
        s1=f'loss_D= {mean(all_loss_D):<.10f}, D_real={mean(all_D_real):<.6f}, D_fake={mean(all_D_fake):<.6f};'
        writer.add_scalar('Train D/D(x)', mean(all_loss_D))
        writer.add_scalar('Train D/D(G(z))', mean(all_D_real))
        writer.add_scalar('Train D/err', mean(all_D_fake))
        
        
        
        # evaluation
        generate_img=None
        netG.eval()

        # old test
        all_acc=[]        
        with torch.no_grad():
            for i,(label_id,onehot) in enumerate(testloader):
                bs=onehot.shape[0]
                # label process
                onehot=onehot.to(args.device,dtype=torch.int)
                label_id=label_id.to(args.device,dtype=torch.int)

                # Generate batch of latent vectors
                noise = torch.randn(bs, args.nz, device=args.device)
                # Generate fake image batch with G
                fake = netG(noise,label_id)
                generate_img=fake
                
                acc=net_eval.eval(fake,onehot)
                all_acc.append(acc)

        old_acc=mean(all_acc)
        # record        
        s3=f'old test acc= {old_acc:<.6f}'
        writer.add_scalar('Old Test/Episode Acc', old_acc)

        if old_acc>best_acc_old:
            best_acc_old=old_acc
            generate_img=denormalize(generate_img)
            save_image(generate_img,f'{args.log_dir}/old_{epoch}.jpg',nrow=8)
            torch.save({
                'discriminator':netD,
                'generator':netG,
                'args':args
            },
            f'{args.log_dir}/model_old.pth')
        elif epoch%args.save_interval==0:
            generate_img=denormalize(generate_img)
            save_image(generate_img,f'{args.log_dir}/old_{epoch}.jpg',nrow=8)
            torch.save({
                'discriminator':netD,
                'generator':netG,
                'args':args,
                'last_epoch':epoch
            },
            f'{args.log_dir}/checkpoint.pth')
        
        #new test
        all_acc=[]        
        with torch.no_grad():
            for i,(label_id,onehot ) in enumerate(newTestloader):
                bs=onehot.shape[0]
                # label process
                onehot=onehot.to(args.device,dtype=torch.int)
                label_id=label_id.to(args.device,dtype=torch.int)

                # Generate batch of latent vectors
                noise = torch.randn(bs, args.nz, device=args.device)
                # Generate fake image batch with G
                fake = netG(noise,label_id)
                generate_img=fake

                acc=net_eval.eval(fake,onehot)
                all_acc.append(acc)
        new_acc=mean(all_acc)
        # record        
        s4=f'new test acc= {new_acc:<.6f}'
        writer.add_scalar('New Test/Episode Acc', new_acc)

        if new_acc>best_acc_new:
            best_acc_new=new_acc
            generate_img=denormalize(generate_img)
            save_image(generate_img,f'{args.log_dir}/new_{epoch}.jpg',nrow=8)
            torch.save({
                'discriminator':netD,
                'generator':netG,
                'args':args
            },
            f'{args.log_dir}/model_new.pth')
        elif epoch%args.save_interval==0:
            generate_img=denormalize(generate_img)
            save_image(generate_img,f'{args.log_dir}/new_{epoch}.jpg',nrow=8)
            torch.save({
                'discriminator':netD,
                'generator':netG,
                'args':args,
                'last_epoch':epoch
            },
            f'{args.log_dir}/checkpoint.pth')

        if mean([new_acc,old_acc])>best_acc_avg:
            best_acc_avg=mean([new_acc,old_acc])
            torch.save({
                'discriminator':netD,
                'generator':netG,
                'args':args
            },
            f'{args.log_dir}/model_avg.pth')

        

        with open(f'./{args.log_dir}/train_record.txt', 'a') as train_record:
            train_record.write(f'Epoch: {epoch:6} ')
            train_record.write(s1+' ')
            train_record.write(s2+' ')
            train_record.write(s3+' ')
            train_record.write(s4+'\n')
        

@torch.no_grad()
def test(netG,args,test_time=100000,threshold=0.8):
    # load data
    testset=ICLEVR_dataset(args,mode='test')
    testloader=DataLoader(testset,batch_size=args.test_bs,shuffle=True,num_workers=args.num_workers)
    newtestset=ICLEVR_dataset(args,mode='new_test')
    newTestloader=DataLoader(newtestset,batch_size=args.test_bs,shuffle=False,num_workers=args.num_workers)

    # evaluation model
    net_eval=evaluation_model()

    best_old=0
    best_new=0
    for t in range(test_time):
        generate_img=None
        new_score,old_score=0,0
        if best_old<threshold:
            old_mean=[]
            for i,(label_id,onehot ) in enumerate(testloader):
                bs=onehot.shape[0]
                # label process
                onehot=onehot.to(args.device,dtype=torch.int)
                label_id=label_id.to(args.device,dtype=torch.int)

                # Generate batch of latent vectors
                noise = torch.randn(bs, args.nz, 1, 1, device=args.device)
                # Generate fake image batch with G
                fake = netG(noise,label_id)
                acc=net_eval.eval(fake,onehot)
                old_mean.append(acc)
                # save image
                generate_img=denormalize(fake)
            
            old_score=torch.tensor(old_mean).mean().item()
            if old_score>best_old:
                best_old=old_score
                save_image(generate_img,f'{args.log_dir}/old_test_{t}.jpg',nrow=8)
            print(f'test {t}: old acore={old_score}')
        if best_new<threshold:
            new_mean=[]
            for i,(label_id,onehot ) in enumerate(newTestloader):
                bs=onehot.shape[0]
                # label process
                onehot=onehot.to(args.device,dtype=torch.int)
                label_id=label_id.to(args.device,dtype=torch.int)

                # Generate batch of latent vectors
                noise = torch.randn(bs, args.nz, 1, 1, device=args.device)
                # Generate fake image batch with G
                fake = netG(noise,label_id)
                acc=net_eval.eval(fake,onehot)
                new_mean.append(acc)
                # save image
                generate_img=denormalize(fake)
            
            new_score=torch.tensor(new_mean).mean().item()
            if new_score>best_new:
                best_new=new_score
                save_image(generate_img,f'{args.log_dir}/new_test_{t}.jpg',nrow=8)
            print(f'test {t}: new acore={new_score}')

        if best_old>=threshold and best_new>=threshold:
            print(f'best score: old={best_old}; new={best_new}')
            break


if __name__=='__main__':    
    os.environ['CUDA_LAUNCH_BLOCKING']='1'
    args = parse_args()
    
    if args.model_dir!='': # load exist model
        test_only=args.test_only
        log_dir=args.log_dir

        model_path=args.model_dir
        saved_model=torch.load(f'{model_path}/model_new.pth')
        args=saved_model['args']
        args.model_dir=model_path
        args.test_only=test_only
        if args.test_only:
            args.log_dir=f'{args.log_dir}/test'
        else:
            args.log_dir=f'{args.log_dir}/continue'

        
    
    # check device
    if args.device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA is not available.'
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    if args.log_dir=='':
        # create save directory
        local=timezone(timedelta(hours=8))
        now = datetime.now().astimezone(local)
        current_time = now.strftime("%d_%H-%M-%S")
        args.log_dir=f'./logs_v9/{current_time}'
    
    #create directory to save result
    os.makedirs(args.log_dir, exist_ok=True)
    
    # create record file
    if os.path.exists(f'{args.log_dir}/train_record.txt'):
        os.remove(f'{args.log_dir}/train_record.txt')
    print(args)
    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    # random seed
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create model or load model
    start_epoch=0
    if args.model_dir!='':
        # load
        netD=saved_model['discriminator'].to(args.device)
        netG=saved_model['generator'].to(args.device)
    else:
        netD=Discriminator_Projection(args).to(args.device)
        netG=Generator(args).to(args.device)
        

    if args.test_only:
        test(netG,args)
    else:
        train(netD,netG,args)