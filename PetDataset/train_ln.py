from dataset import Pascal_VOC_LN
from network import Resnet_LN
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
import torch

data_dir = 'pet_dataset/'
test_data_dir = 'pet_dataset/'

logger = TensorBoardLogger("logger", name= "my_model_hu")
# profiler = PyTorchProfiler(
#     on_trace_ready= torch.profiler.tensorboard_trace_handler("logger/profiler0"),
#     schedule= torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active= 20),
# )

dm = Pascal_VOC_LN(data_dir, test_data_dir, batch_size= 32, num_workers= 11)
model = Resnet_LN(num_classes= 37).cuda()

trainer = pl.Trainer(accelerator= "gpu", devices= [0,1,2,3], min_epochs=1, max_epochs= 10, precision= 16, 
                     callbacks= EarlyStopping(monitor= 'val_loss'), logger= logger
                     )
trainer.fit(model, dm)
trainer.validate(model, dm)
