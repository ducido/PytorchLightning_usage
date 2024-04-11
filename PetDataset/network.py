import torchvision.models as models
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
import torch
import torchmetrics
import torchvision

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
            # if int(name[-1]) == 4:
            #     continue
            layer = getattr(self.model, name)
            for param in layer.parameters():
                param.require_grad = False

        self.model.fc = nn.Sequential(
            nn.Dropout(p= 0.2),
            nn.Linear(in_features= 2048, out_features= self.num_classes),
            nn.Softmax()
        )
            
    def forward(self, x):
        return self.model(self.transform(x))

class MyAccuracy(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("true_positive", default= torch.tensor(0), dist_reduce_fx= 'sum')
    
    def update(self, pred, true):
        pred = torch.argmax(pred, dim= -1)
        assert pred.shape == true.shape
        self.true_positive = (pred == true).sum()
        self.total_sample = len(pred)
    
    def compute(self):
        return self.true_positive / self.total_sample

class Resnet_LN(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes= num_classes
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights= weights)
        self.transform = weights.transforms()
        self.accuracy = MyAccuracy() 

        self.config_model()

    def config_model(self):
        layer_names = [f'layer{i}' for i in range(1,5)]

        for name in layer_names:
            if int(name[-1]) >= 3:
                continue
            layer = getattr(self.model, name)
            for param in layer.parameters():
                param.require_grad = False

        self.model.fc = nn.Sequential(
            nn.Dropout(p= 0.2),
            nn.Linear(in_features= 2048, out_features= self.num_classes),
            nn.Softmax(dim= -1)
        )
            
    def forward(self, x):
        return self.model(self.transform(x))
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = F.cross_entropy(pred, y)
        self.log_dict({'train_loss': loss}, on_epoch= True, prog_bar= True)

        if batch_idx % 10 == 0:
            x = x[:4]
            grid = torchvision.utils.make_grid(x)
            self.logger.experiment.add_image("some images", grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = F.cross_entropy(pred, y)
        acc = self.accuracy(pred, y)
        self.log_dict({'val_acc': acc, 'val_loss': loss}, on_epoch= True, prog_bar= True)
        return acc
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr= 1e-3)