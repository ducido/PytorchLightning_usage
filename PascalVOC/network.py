import torchvision.models as models
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
import torch
import torchmetrics

class Resnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights= weights)
        self.transform = weights.transforms()
        self.config_model()

    def config_model(self):
        layer_names = [f'layer{i}' for i in range(1,5)]

        for name in layer_names:
            if int(name[-1]) == 4:
                continue
            layer = getattr(self.model, name)
            for param in layer.parameters():
                param.require_grad = False

        self.model.fc = nn.Sequential(
            nn.Dropout(p= 0.2),
            nn.Linear(in_features= 2048, out_features= 20),
            nn.Sigmoid()
        )
            
    def forward(self, x):
        return self.model(self.transform(x))

class MyAccuracy(torchmetrics.Metric):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.add_state("acc_on_each_sample", default= torch.tensor(0), dist_reduce_fx= 'sum')
        self.add_state("acc_on_batch", default= torch.tensor(0), dist_reduce_fx= 'sum')
    
    def update(self, pred, true):
        pred = torch.round(pred)
        assert pred.shape == true.shape
        self.acc_on_each_sample = (pred == true).sum(axis= -1) / self.num_labels
    
    def compute(self):
        return self.acc_on_each_sample.sum() / len(self.acc_on_each_sample)

class Resnet_LN(pl.LightningModule):
    def __init__(self, num_labels):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights= weights)
        self.transform = weights.transforms()
        self.accuracy = torchmetrics.Accuracy(task= 'multilabel', num_labels= num_labels) 
        self.accuracy = MyAccuracy(num_labels= num_labels) 

        self.config_model()

    def config_model(self):
        layer_names = [f'layer{i}' for i in range(1,5)]

        for name in layer_names:
            if int(name[-1]) == 4:
                continue
            layer = getattr(self.model, name)
            for param in layer.parameters():
                param.require_grad = False

        self.model.fc = nn.Sequential(
            nn.Dropout(p= 0.2),
            nn.Linear(in_features= 2048, out_features= 20),
            nn.Sigmoid()
        )
            
    def forward(self, x):
        return self.model(self.transform(x))
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        self.log_dict({'train_loss': loss}, on_epoch= True, prog_bar= True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        acc = self.accuracy(pred, y)
        self.log_dict({'val_acc': acc, 'val_loss': loss}, on_epoch= True, prog_bar= True)
        return acc
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr= 1e-3)
    
if __name__ == '__main__':
    model = Resnet(num_classes= 20).cuda()