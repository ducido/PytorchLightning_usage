from dataset import Pascal_VOC_LN
from network import Resnet_LN
import pytorch_lightning as pl

data_dir = '/media/minhduc/D/AAAAA/Pixta/archive/voctrainval_06-nov-2007/VOCdevkit/VOC2007/'

dm = Pascal_VOC_LN(data_dir, batch_size= 32, num_workers= 19)
model = Resnet_LN(num_labels= 20).cuda()

trainer = pl.Trainer(accelerator= "gpu", devices= [0], min_epochs=1, max_epochs= 3, precision= 16)
trainer.fit(model, dm)
trainer.validate(model, dm)
