from dataset import Pascal_VOC_LN
from network import Resnet_LN
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
import torch

data_dir = '/media/minhduc/D/AAAAA/Pixta/PascalVOC/archive/voctrainval_06-nov-2007/VOCdevkit/VOC2007/'

logger = TensorBoardLogger("logger", name= "my_model_hu")
profiler = PyTorchProfiler(
    on_trace_ready= torch.profiler.tensorboard_trace_handler("logger/profiler0"),
    schedule= torch.profiler.schedule(skip_first=20, wait=1, warmup=1, active= 20)
)

dm = Pascal_VOC_LN(data_dir, batch_size= 32, num_workers= 19)
model = Resnet_LN(num_labels= 20).cuda()

trainer = pl.Trainer(accelerator= "gpu", devices= [0], min_epochs=1, max_epochs= 10, precision= 16, 
                     callbacks= EarlyStopping(monitor= 'val_loss'), logger= logger, profiler= profiler)
trainer.fit(model, dm)
trainer.validate(model, dm)
