import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import variable
from torchvision import transforms, models, datasets
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import glob
import math
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm
import math
import datetime as dt
from os.path import exists
import shutil
import pickle
from PIL import ImageGrab, Image

condition = 'Lwetground'

def mat_mul(A, B):
    return torch.matmul(A, B)

def count_parameters(model):
     return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class textcolors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    DEFAULT = "\033[37m"
    ENDC = '\033[0m'
    
def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def loss_fn(preds, label, mu, logvar):
    # MSE = F.mse_loss(preds, label, reduction='sum')
    MSE = F.smooth_l1_loss(preds, label, reduction='sum')
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + kld_loss, MSE, kld_loss


class FeatureReshaper(nn.Module):
    def __init__(self, input_size):
        super(FeatureReshaper, self).__init__()
        self.fc = nn.Linear(input_size, 784)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 28, 28)  # Reshape to [batch_size, 1, 28, 28]
        return x




class PointFusion(nn.Module):
    def __init__(self, device):
        super(PointFusion,self).__init__()

        h_dim = 12
        z_dim = 3
        self.float()
        
        self.point_transform_net = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=True),
            nn.PReLU(),
            nn.Conv1d(64, 128, 1, bias=True),
            nn.PReLU(),
            nn.Conv1d(128, 1024, 1, bias=True),
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=2048),
            nn.Flatten(),
            nn.Linear(1024, 512, bias=True),
            nn.PReLU(),
            nn.Linear(512, 256, bias=True),
            nn.PReLU(),
            nn.Linear(256, 9, bias=True)
        ).to(device)

        self.point_transform_forward = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(3, 64, 1, bias=True),
            nn.PReLU(),
            nn.Conv1d(64, 64, 1, bias=True),
            nn.PReLU()
        ).to(device)

        self.feature_transform_net = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=True),
            nn.PReLU(),
            nn.Conv1d(64, 128, 1, bias=True),
            nn.PReLU(),
            nn.Conv1d(128, 1024, 1, bias=True),
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=2048),
            nn.Flatten(),
            nn.Linear(1024, 512, bias=True),
            nn.PReLU(),
            nn.Linear(512, 256, bias=True),
            nn.PReLU(),
            nn.Linear(256, 64*64, bias=True)
        ).to(device)

        self.feature_transform_forward = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(64, 64, 1, bias= True),
            nn.PReLU(),
            nn.Conv1d(64, 128, 1, bias=True),
            nn.PReLU(),
            nn.Conv1d(128, 1024, 1, bias=True),
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=2048),
            nn.Flatten()
        ).to(device)

        self.fused_encoder = nn.Sequential(
            nn.PReLU(),
            nn.Linear(3072, 1536, bias=True),
            nn.PReLU(),
            nn.Linear(1536, 768, bias=True),
            nn.PReLU(),
            nn.Linear(768, 384, bias=True),
            nn.PReLU(),
            nn.Linear(384, 192, bias=True),
            nn.PReLU(),
            nn.Linear(192, 96, bias=True),
            nn.PReLU(),
            nn.Linear(96, 48, bias=True),
            nn.PReLU(),
            nn.Linear(48, 24, bias=True),
            nn.PReLU(),
            nn.Linear(24, h_dim, bias=True),
            nn.PReLU()
        ).to(device)

        self.lat_mu = nn.Linear(h_dim,z_dim).to(device)
        self.lat_logvar = nn.Linear(h_dim,z_dim).to(device)
        self.sample = nn.Linear(z_dim,h_dim).to(device)

        self.box_decoder = nn.Sequential(
            nn.PReLU(),
            nn.Linear(h_dim + 4, 32, bias=True),
            nn.PReLU(),
            nn.Linear(32, 64, bias=True),
            nn.PReLU(),
            nn.Linear(64, 128, bias=True),
            nn.PReLU(),
            nn.Linear(128, 256, bias=True),
            nn.PReLU(),
            nn.Linear(256, 512, bias=True),
            nn.PReLU(),
            nn.Linear(512, 1024, bias=True),
            nn.PReLU(),
            nn.Linear(1024, 1024, bias=True),
            nn.PReLU(),
            nn.Linear(1024, 2048, bias=True),
            nn.PReLU(),
        ).to(device)

        self.box = nn.Linear(2048, 24, bias=True).to(device)

        init_modules = [self.point_transform_net, self.point_transform_forward, self.feature_transform_net, self.feature_transform_forward,
                        self.lat_mu, self.lat_logvar, self.sample,self.fused_encoder,
                        self.box_decoder,self.box]
        
        for module in init_modules:
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(device)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def forward(self, x_point, x_image, x_2dbox):
        a = self.point_transform_net(x_point)
        input_T = a.view(x_point.shape[0],3,3)
        g = mat_mul(x_point.transpose(1,2),input_T)
        g = self.point_transform_forward(g.transpose(2,1))

        b = self.feature_transform_net(g)
        feature_T = b.view(x_point.shape[0],64,64)
        g = mat_mul(g.transpose(1,2),feature_T)
        lidar_feature = self.feature_transform_forward(g.transpose(2,1))
        # Extracting features for points
        point_features = self.feature_transform_forward(g.transpose(2,1))

        # Extracting features for images (assuming x_image is already a feature representation)
        image_features = x_image

        # print(x_image.shape, lidar_feature.shape)
        fused_feature = torch.hstack((x_image,lidar_feature))
        fused_encoded = self.fused_encoder(fused_feature)
        enc_feat = self.fused_encoder(fused_feature)

        fused_mu = self.lat_mu(fused_encoded)
        fused_logvar = self.lat_logvar(fused_encoded)

        fused_z = self.reparameterize(fused_mu,fused_logvar)
        
        fused_sample = self.sample(fused_z)
        box_sample = torch.hstack((fused_sample,x_2dbox))

        box_preds = self.box_decoder(box_sample)

        boxes = self.box(box_preds)
        #print("Model->boxes-", boxes.shape)
        #print(boxes)

        return fused_mu, fused_logvar, boxes, enc_feat #point_features, image_features

class textcolors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    DEFAULT = "\033[m"

set_seeds(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

points =  np.load('train_points_pt_'+condition+'.npy')
labels = np.load('train_labels_pt_'+condition+'.npy')
boxes2d = np.load('train_boxes2d_pt_'+condition+'.npy')
intermediate_output = np.load('intermediate_output_pt_'+condition+'.npy')
# print("potints-", points.shape)
# print("labels-", labels.shape)
# print("boxes2d-", boxes2d.shape)
# print("intermediate-output-", intermediate_output.shape)



# print the model summary
model = PointFusion(device=device)
print("Model Summary (Trainable Parameters = {}):\n".format(count_parameters(model)))
#print(model)

# define optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
optimizer_to(optimizer,device)

# decide on train/val ratio
train_points = points#[:x]
test_points = points#[x:]

train_boxes2d = boxes2d
test_boxes2d = boxes2d

train_labels = labels
test_labels = labels

train_intermediate = intermediate_output
test_intermediate = intermediate_output

#epoch number
epo = 100

train_bs = 32
val_bs = 4

perm1 = np.random.permutation(len(train_points))
# print("perm1-",perm1)
train_points = train_points[perm1]
train_labels = train_labels[perm1]
train_boxes2d = train_boxes2d[perm1]
train_intermediate = train_intermediate[perm1]

# Prepare the training dataset.
point_train_batches = DataLoader(train_points, batch_size=train_bs)
image_train_batches = DataLoader(train_intermediate,batch_size=train_bs)
label_train_batches = DataLoader(train_labels, batch_size=train_bs)
boxes_train_batches = DataLoader(train_boxes2d,batch_size=train_bs)
#classes_train_batches = DataLoader(train_classes, batch_size=train_bs)

point_val_batches = DataLoader(test_points, batch_size=val_bs)
image_val_batches = DataLoader(test_intermediate,batch_size=val_bs)
label_val_batches = DataLoader(test_labels, batch_size=val_bs)
boxes_val_batches = DataLoader(test_boxes2d,batch_size=val_bs)
#classes_val_batches = DataLoader(dev_classes, batch_size=val_bs)

min_validation_loss = np.inf
count = 0
num_epoch = []
train_losses = []
valid_losses = []
wait = 50



for epoch in range(1, epo + 1):
    # all_point_features = []
    # all_image_features = []
    enc_feat_list = []


    print("")
    start_time = time.time()

    avg_train_loss = 0.0
    avg_val_loss = 0.0
    avg_train_box_mse = 0.0
    avg_train_box_kld = 0.0
    avg_val_box_mse = 0.0
    avg_val_box_kld = 0.0
    #avg_class_train_loss = 0.0
    #avg_class_val_loss = 0.0
    
    model.train()

    data = zip(point_train_batches,image_train_batches,label_train_batches,boxes_train_batches)
    progressbar = tqdm(data, unit = ' Batch', total=len(point_train_batches))
    for step, (x_points,x_images,y_boxes3d,y_boxes2d) in enumerate(progressbar):
        x_images = x_images.to(device)
        x_points = x_points.to(device)
        y_boxes3d = y_boxes3d.to(device)
        y_boxes2d = y_boxes2d.to(device)
                
        #mu, logvar, boxes, point_features, image_features = model(x_points,x_images,y_boxes2d)
        mu, logvar, boxes, enc_feat = model(x_points,x_images,y_boxes2d)
        #print(enc_feat)

        # Assuming point_features has shape [batch_size, feature_size]
        # feature_size_p = point_features.size(1)
        # reshaper_p = FeatureReshaper(feature_size_p).to(device)  # Move to the same device as your data
        # feature_size_i = image_features.size(1)
        # reshaper_i = FeatureReshaper(feature_size_i).to(device)
        feature_size = enc_feat.size(1)
        reshaper = FeatureReshaper(feature_size).to(device)  # Move to the same device as your data
        # Convert point_features
        # image_like_features_p = reshaper_p(point_features)
        # image_like_features_i = reshaper_i(image_features)
        image_like_features = reshaper(enc_feat)



        # all_point_features.append(image_like_features_p.cpu().detach())
        # all_image_features.append(image_like_features_i.cpu().detach())
        enc_feat_list.append(image_like_features.cpu().detach())

        # Properly reshaping
        tmp = torch.empty(size=(boxes.shape[0], 8, 3)).to(device) 
        for i in range(boxes.shape[0]):
            t2 = boxes[i]
            t2_2d = torch.reshape(t2, shape=(8, 3))
            # final = torch.cat([final, t2_2d], dim=0)
            tmp[i] = t2_2d
        # print("shape-", tmp.shape, tmp.device, device)

        
        total_loss,box_mse,box_kld = loss_fn(tmp, y_boxes3d, mu, logvar)

        # print(y_boxes3d.dtype)

        
        total_loss = total_loss.float()
        box_mse = box_mse.float()  
        # print(total_loss.dtype, box_mse.dtype, box_kld.dtype)
        # print(total_loss)


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        avg_train_loss += total_loss.item()
        avg_train_box_mse += box_mse
        avg_train_box_kld += box_kld

        progressbar.set_description(textcolors.DEFAULT + "Epoch [{}/{}] | Training Batch Loss = {:.4f}".format(epoch,epo,total_loss.item()))
    if epoch == epo:  # Check if it's the last epoch
        # Concatenate all features
        # all_point_features = torch.cat(all_point_features, dim=0)
        # all_image_features = torch.cat(all_image_features, dim=0)
        enc_feat_list = torch.cat(enc_feat_list, dim=0)

        # Save the concatenated features
        # torch.save(all_point_features, 'all_point_features.pt')
        # torch.save(all_image_features, 'all_image_features.pt')
        torch.save(enc_feat_list, 'enc_feat_'+condition+'.pt')
    model.eval()
    data = zip(point_val_batches,image_val_batches,label_val_batches,boxes_val_batches)
    progressbar = tqdm(data, unit = ' Batch', total=len(point_val_batches))
    for step, (x_points,x_images,y_boxes3d,y_boxes2d) in enumerate(progressbar):
        
        x_images = x_images.to(device)
        x_points = x_points.to(device)
        y_boxes3d = y_boxes3d.to(device)
        y_boxes2d = y_boxes2d.to(device)

        # print(f"{textcolors.RED}line 361 - {x_points.shape}-{x_images.shape}-{y_boxes2d.shape}{textcolors.DEFAULT}")
     
        with torch.no_grad():   
            #mu, logvar, boxes, point_features, image_features = model(x_points,x_images,y_boxes2d)
            mu, logvar, boxes, enc_feat = model(x_points,x_images,y_boxes2d)

        tmp = torch.empty(size=(boxes.shape[0], 8, 3)).to(device) 
        for i in range(boxes.shape[0]):
            t2 = boxes[i]
            t2_2d = torch.reshape(t2, shape=(8, 3))
            # final = torch.cat([final, t2_2d], dim=0)
            tmp[i] = t2_2d
        total_loss,box_mse,box_kld = loss_fn(tmp, y_boxes3d, mu, logvar)

        avg_val_loss += total_loss.item()
        avg_val_box_mse += box_mse
        avg_val_box_kld += box_kld

        progressbar.set_description(textcolors.DEFAULT + "Epoch [{}/{}] | Validation Batch Loss = {:.4f}".format(epoch,epo,total_loss.item()))

    avg_train_loss = avg_train_loss / len(point_train_batches)
    avg_train_box_mse = avg_train_box_mse / len(point_train_batches)
    avg_train_box_kld = avg_train_box_kld / len(point_train_batches)
    avg_val_loss = avg_val_loss / len(point_val_batches)
    avg_val_box_mse = avg_val_box_mse / len(point_val_batches)
    avg_val_box_kld = avg_val_box_kld / len(point_val_batches)
    

    train_box = [torch.round(avg_train_box_mse, decimals=3),torch.round(avg_train_box_kld,decimals=3)]
    val_box = [torch.round(avg_val_box_mse,decimals=3),torch.round(avg_val_box_kld,decimals=3)]

    num_epoch.append(epoch)
    train_losses.append(avg_train_loss)
    valid_losses.append(avg_val_loss)
    
    # print(textcolors.BLUE + "End of epoch: {} | Stop Count: [{}/{}] | Time taken: {:.2f}".format(epoch,count,wait,time.time() - start_time))
    
    # print(textcolors.YELLOW + "Box Resid: {}".format(((y_boxes3d[0] - boxes[0])/y_boxes3d[0])*100))
    print("Truth Box: {}".format(y_boxes3d[0]))
    print("Pred Box: {}".format(boxes[0]))
    print(textcolors.GREEN + "(Average training) Total Loss: {:.3f} | Box MSE,KLD: {}".format(avg_train_loss,train_box))
    print("(Average validation) Total Loss: {:.3f} | Box MSE,KLD: {}".format(avg_val_loss,val_box))

    if min_validation_loss > avg_val_loss:
        torch.save(model.state_dict(), 'lowest_val_loss_'+condition+'.pt')
        print(textcolors.YELLOW + "New minimum validation loss--model saved!")
        min_validation_loss = avg_val_loss
        count = 0
    else:
        count += 1
        if count == wait:
            print(textcolors.RED + "Minimum validation loss has not changed for {} epochs. Exiting training now.".format(wait))
            break

fig = plt.figure()
plt.plot(num_epoch,train_losses,'-b',label='Training Loss')
plt.plot(num_epoch,valid_losses,'-r',label='Validation Loss')
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Average Training and Validation Combined Loss, Batch Size: {}".format(train_bs))
plt.legend()
plt.savefig('loss.png',bbox_inches='tight',dpi=300)

