import torch
from torch.nn import functional as F
from dataset import PascalVOC
from torchvision import transforms
from network import Resnet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # Import the SummaryWriter

num_epochs = 20
data_dir = 'pet_dataset/'
test_data_dir = 'pet_dataset/'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((240,240)),
])
train_dataset = PascalVOC(data_dir, types='trainval', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = PascalVOC(test_data_dir, types= 'test', transform= transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle= False)

model = Resnet(num_classes=37).cuda()
optim = torch.optim.Adam(params=model.parameters(), lr=1e-3)

# Initialize the TensorBoard writer
writer = SummaryWriter()

def train_step(model, x, y):
    model.train()
    x, y = x.cuda(), y.cuda()
    pred = model(x)
    loss = F.cross_entropy(pred, y)

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss

def val_step(model, x, y):
    model.eval()
    x, y = x.cuda(), y.cuda()

    with torch.no_grad():
        pred = model(x)
    pred = torch.argmax(pred, dim= -1)
    acc = (pred == y).sum() / len(pred)
    return acc

for epoch in range(num_epochs):
    # Training phase
    train_loss = 0
    train_loss_avg = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - training")
    for batch_idx, (x, y) in enumerate(train_bar):
        train_loss = train_step(model, x, y)
        train_loss_avg = (train_loss + batch_idx * train_loss_avg) / (batch_idx+1)
        train_bar.set_postfix(train_loss_average=train_loss_avg.item())

    # Log training loss
    writer.add_scalar("Loss/train", train_loss.item(), epoch)

    # Validation phase
    val_acc = 0
    val_acc_avg = 0
    val_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - validation")
    for batch_idx, (x, y) in enumerate(val_bar):
        val_acc = val_step(model, x, y)
        val_acc_avg = (val_acc + batch_idx * val_acc_avg) / (batch_idx + 1)
        val_bar.set_postfix(val_acc_average=val_acc_avg.item())

    # Log validation accuracy
    writer.add_scalar("Accuracy/val", val_acc.item(), epoch)

# Close the TensorBoard writer after training is complete
writer.close()
