import torch
from torch.nn import functional as F
from dataset import PascalVOC
from torchvision import transforms
from network import Resnet
from tqdm import tqdm

num_epochs = 20
data_dir = '/media/minhduc/D/AAAAA/Pixta/archive/voctrainval_06-nov-2007/VOCdevkit/VOC2007/'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
])
train_dataset = PascalVOC(data_dir, types= 'trainval', transform= transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= 32, shuffle= True)
val_dataset = PascalVOC(data_dir, types= 'train', transform= transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size= 32, shuffle= False)

model = Resnet(num_classes= 20).cuda()
optim = torch.optim.Adam(params= model.parameters(), lr= 1e-3)

def train_step(model, x, y):
    model.train()
    x, y = x.cuda(), y.cuda()
    pred = model(x)
    loss = F.binary_cross_entropy(pred, y)

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss

def val_step(model, x, y):
    model.eval()
    x, y = x.cuda(), y.cuda()

    pred = model(x)
    pred = torch.round(pred)
    acc = ((pred == y).sum(axis= 0) / 20).sum() / len(pred)
    return acc


for epoch in range(num_epochs):
    # Training phase
    train_loss = 0
    train_loss_avg = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - training")
    for batch_idx, (x, y) in enumerate(train_bar):
        pre = batch_idx * train_loss_avg
        train_loss = train_step(model, x, y)
        train_loss_avg = (train_loss + pre) / (batch_idx+1)
        train_bar.set_postfix(train_loss_avg = train_loss_avg.item())

    # Validation phase
    val_acc = 0
    val_acc_avg = 0
    var_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - validation")
    for batch_idx, (x, y) in enumerate(var_bar):
        pre = batch_idx * val_acc_avg
        val_acc = val_step(model, x, y)
        val_acc_avg = (val_acc + pre) / (batch_idx + 1)
        var_bar.set_postfix(val_acc_avg = val_acc_avg.item())