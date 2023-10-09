import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import DCGAN_VAE_pixel_v2 as DVAE
import torch.nn.functional as F
import copy
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn import metrics


condition = 'Hfog'
iter = 5 ## choose from this list: [5, 10, 25, 50, 75, 100]
GD_method = 'ZO_SGD' ## choose between 'ZO_Sign', 'ZO_SGD', and 'SPSA'

def SPSA_gradient(model, loss_function, input_data, target, c=1e-3):
    """
    Compute the gradient approximation using Simultaneous Perturbation Stochastic Approximation (SPSA).
    """
    # Generate a random direction (delta)
    delta = torch.tensor([1.0 if np.random.rand() > 0.5 else -1.0 for _ in model.parameters()], requires_grad=True).to(device)
    
    # Perturb the model parameters positively
    for param, d in zip(model.parameters(), delta):
        param.data += c * d

    # Compute loss for positively perturbed model
    loss_positive = loss_function(model, input_data, target)

    # Perturb the model parameters negatively
    for param, d in zip(model.parameters(), delta):
        param.data -= 2 * c * d

    # Compute loss for negatively perturbed model
    loss_negative = loss_function(model, input_data, target)

    # Compute the gradient approximation
    gradient_approximation = [(loss_positive - loss_negative) / (2 * c * d) for d in delta]

    # Revert the model parameters to their original values
    for param, d in zip(model.parameters(), delta):
        param.data += c * d

    return gradient_approximation

def ZO_Sign_gradient_approximation(model, loss_function, input_data, target, epsilon=1e-6):
    """
    Approximate the gradient using Zeroth-Order Stochastic Gradient Descent with Sign (ZO-Sign).
    """
    original_loss = loss_function(model, input_data, target)
    gradient_approximations = []

    for param in model.parameters():
        perturbation = torch.randn_like(param, requires_grad=True).to(device)
        param.data += epsilon * perturbation

        perturbed_loss = loss_function(model, input_data, target)
        gradient_approximation = torch.sign((perturbed_loss - original_loss) / epsilon) * perturbation
        gradient_approximations.append(gradient_approximation)

        # Revert the parameter to its original value
        param.data -= epsilon * perturbation

    return gradient_approximations

def ZO_SGD_gradient_approximation(model, loss_function, input_data, target, sigma):
    """
    Approximate the gradient using Zeroth-Order Stochastic Gradient Descent (ZO-SGD).
    """
    original_loss = loss_function(model, input_data, target)
    gradient_approximations = []

    for param in model.parameters():
        epsilon = torch.randn_like(param, requires_grad=True).to(device)
        param.data += sigma * epsilon

        perturbed_loss = loss_function(model, input_data, target)
        gradient_approximation = -(perturbed_loss - original_loss) / sigma * epsilon
        gradient_approximations.append(gradient_approximation)

        # Revert the parameter to its original value
        param.data -= sigma * epsilon

    return gradient_approximations

def loss_function(model, input_data, target):
    b = input_data.size(0)
    [z, mu, logvar] = model(input_data)
    recon = netG(z)
    recon = recon.contiguous()
    recon = recon.view(-1, 256)
    ## for features
    recon = torch.argmax(recon, dim=1)
    recon = recon[:len(target)]
    target = target.float()
    recon = recon.float()
    # print("Shape of recon:", recon.shape)
    # print("Shape of target:", target.shape)

    ## for features
    recl = loss_fn(recon, target)
    recl = torch.sum(recl) / b
    kld = KL_div(mu, logvar)
    loss = recl + kld.mean()
    # if regularization is not None:
    #     loss += regularization

    return loss


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return (img,index)

def KL_div(mu,logvar,reduction = 'none'):
    mu = mu.view(mu.size(0),mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    
    if reduction == 'sum':
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1) 
        return KL

def store_NLL(x, recon, mu, logvar, z):
    with torch.no_grad():
        sigma = torch.exp(0.5*logvar)
        b = x.size(0)
        target = Variable(x.data.view(-1) * 255).long()
        recon = recon.contiguous()
        recon = recon.view(-1,256)
        recon = torch.argmax(recon, dim=1)
        recon = recon[:len(target)]
        target = target.float()
        recon = recon.float()
        #print("shape of recon:", recon.shape)
        #print("shape of target:", target.shape)


        cross_entropy = F.cross_entropy(recon, target, reduction='none')
        #print("shape of cross_entropy:", cross_entropy)
        log_p_x_z = -torch.sum(cross_entropy.view(b ,-1), 1)
      
        log_p_z = -torch.sum(z**2/2+np.log(2*np.pi)/2,1)
        z_eps = (z - mu)/sigma
        #print("z_eps shape:",z_eps.shape)
        #z_eps = z_eps.view(opt.repeat,-1)
        z_eps = z_eps.view(100,-1)
        log_q_z_x = -torch.sum(z_eps**2/2 + np.log(2*np.pi)/2 + logvar/2, 1)
        
        weights = log_p_x_z+log_p_z-log_q_z_x
        
    return weights

def compute_NLL(weights):
    
    with torch.no_grad():
        NLL_loss = -(torch.log(torch.mean(torch.exp(weights - weights.max())))+weights.max()) 
        
    return NLL_loss


def adjust_checkpoint_weights(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint

    # Adjust the weights for the final convolutional transpose layer
    old_weight = state_dict['main.final:32-1:convt.weight']
    new_weight = old_weight[:, :, 1:4, 1:4]  # Crop the 4x4 weights to 3x3
    state_dict['main.final:32-1:convt.weight'] = new_weight

    # Save the adjusted checkpoint
    torch.save(checkpoint, 'adjusted_checkpoint.pth')




def compute_regret(dataloader, netE, netG, num_sample, GD_method, KL = "True"):
    
    
    NLL_regret = []
    NLL = []

    for i, x_l in enumerate(dataloader):
        x = x_l[0].to(device)
        weights_agg  = []
        with torch.no_grad():
            for batch_number in range(5):
                
                x = x.to(device)
                b = x.size(0)
                [z,mu,logvar] = netE(x)
                recon = netG(z)
                mu = mu.view(mu.size(0),mu.size(1))
                logvar = logvar.view(logvar.size(0), logvar.size(1))
                z = z.view(z.size(0),z.size(1))
                weights = store_NLL(x, recon, mu, logvar, z)
                weights_agg.append(weights)
            
            weights_agg = torch.stack(weights_agg).view(-1) 
            
            if KL == "True":     
                KL_before = KL_div(mu,logvar,reduction = 'none')
                NLL = np.append(NLL, KL_before.detach().cpu().numpy())
            else:
                NLL_loss_before = compute_NLL(weights_agg)
                NLL= np.append(NLL, NLL_loss_before.detach().cpu().numpy())

        x = x.to(device)
        b = x.size(0)
        netE_copy = copy.deepcopy(netE)
        netE_copy.eval()
        netE_copy1 = copy.deepcopy(netE)
        netE_copy1.eval()

        # ZO-SGD parameters
        sigma = 1e-3  # Perturbation magnitude
        learning_rate = opt.lr
        target = Variable(x.data.view(-1) * 255).long()

        
        for it in range(opt.num_iter):
            if GD_method == 'ZO_Sign':
                gradients = ZO_Sign_gradient_approximation(netE_copy, loss_function, x, target, epsilon=1e-6)
            if GD_method == 'ZO_SGD':
                gradients = ZO_SGD_gradient_approximation(netE_copy, loss_function, x, target, sigma)
            if GD_method == 'SPSA':
                gradients = SPSA_gradient(netE_copy, loss_function, x, target, c=1e-6)
            with torch.no_grad():
                for param, gradient in zip(netE_copy.parameters(), gradients):
                    param.data -= learning_rate * gradient
            
            
        weights_agg  = []
        with torch.no_grad():
            target = Variable(x.data.view(-1) * 255).long()
            for batch_number in range(32):
                [z1,mu1,logvar1] = netE_copy1(x)
                recon1 = netG(z1)
                recon1 = recon1.contiguous()
                mu1 = mu1.view(mu1.size(0),mu1.size(1))
                logvar1 = logvar1.view(logvar1.size(0), logvar1.size(1))
                z1 = z1.view(z1.size(0),z1.size(1))
                weights = store_NLL(x, recon1, mu1, logvar1, z1)
                
                weights_agg.append(weights)
    
            weights_agg = torch.stack(weights_agg).view(-1) 

            if KL == "True":
                KL_after =  KL_div(mu,logvar)
                #print('image {} OPT: {} VAE: {} diff:{}'.format(i, KL_after.item(), KL_before.item(), abs(KL_before.item()  - KL_after.item())))
                regret = abs(KL_before - KL_after)
            else:
                NLL_loss_after = compute_NLL(weights_agg)
                #print('image {} OPT: {} VAE: {} diff:{}'.format(i, NLL_loss_after.item(), NLL_loss_before.item(), abs(NLL_loss_before.item()  - NLL_loss_after.item())))
                regret = abs(NLL_loss_before  - NLL_loss_after)

            NLL_regret = np.append(NLL_regret, regret.detach().cpu().numpy())
        if i >= num_sample: #test for num_sample samples
            break
                
           

    
    return NLL, NLL_regret




parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='./data', help='path to dataset')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=1, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--num_iter', type=int, default=iter, help='number of iters to optimize')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

parser.add_argument('--repeat', type=int, default=200, help='repeat for comute IWAE bounds')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')

parser.add_argument('--state_E', default='./netE_kitti_LDfeat.pth', help='path to encoder checkpoint')
parser.add_argument('--state_G', default='./netG_kitti_LDfeat.pth', help='path to decoder checkpoint')

opt = parser.parse_args()

cudnn.benchmark = True
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)



# Load the extracted features
all_point_features_clear = torch.load('LDfeat_clear.pt')
all_point_features_Hfog = torch.load('LDfeat_'+ condition +'.pt')

feature_dataset = torch.utils.data.TensorDataset(all_point_features_clear)
dataloader_feature = torch.utils.data.DataLoader(feature_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers))
ood_dataset = torch.utils.data.TensorDataset(all_point_features_Hfog)
dataloader_odd = torch.utils.data.DataLoader(ood_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers))


ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
nc = int(opt.nc)

print('Building models...')




netG = DVAE.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu)
state_G = torch.load(opt.state_G, map_location = device)
netG.load_state_dict(state_G)


netE = DVAE.Encoder(opt.imageSize, nz, nc, ngf, ngpu)
state_E = torch.load(opt.state_E, map_location = device)
netE.load_state_dict(state_E)


netG.to(device)
netG.eval()
netE.to(device)
netE.eval()

loss_fn = nn.CrossEntropyLoss(reduction = 'none')

print('Building complete...')


AUC_list = []


for i in range(20):
    ### IN-distribution
    NLL_indist, NLL_regret_indist = compute_regret(dataloader_feature, netE, netG, num_sample=100, GD_method=GD_method, KL = "False")
    ### OOD    
    NLL_ood, NLL_regret_ood = compute_regret(dataloader_odd, netE, netG, num_sample=100, GD_method=GD_method, KL = "False")

    ## AUC calculations
    combined = np.concatenate((NLL_regret_indist, NLL_regret_ood))
    label_1 = np.ones(len(NLL_regret_indist))
    label_2 = np.zeros(len(NLL_regret_ood))
    label = np.concatenate((label_1, label_2))
    fpr, tpr, thresholds = metrics.roc_curve(label, combined, pos_label=0)
    rocauc = metrics.auc(fpr, tpr)

    AUC_list.append(rocauc)
    print('number of runs:', i)


print('For '+GD_method+' with '+str(iter)+' iterations Mean of AUC is:', np.mean(AUC_list))
print('For '+GD_method+' with '+str(iter)+' iterations Variance of AUC is:', np.var(AUC_list))



