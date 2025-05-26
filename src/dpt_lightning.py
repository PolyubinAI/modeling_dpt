
import pytorch_lightning as light
from transformers import DPTForDepthEstimation
from predict_utils import make_prediction_dpt
from utils import load_object
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
LOG_EVERY_N_STEPS = 10

SEED_VALUE = 42

class LightDPT(light.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = load_object(config.model_name)()
        self.model = self.model.train()
        self.criterion = load_object(config.losses[0].loss_fn)()
        self._config = config
        self.mse = load_object(config.losses[1].loss_fn)()
        self.metrics = {i.split('.')[-1]: load_object(i)() for i in config.metrics}

    def forward(self, image):
        return self.model(image).predicted_depth

    def configure_optimizers(self):
        optimizer = load_object(self._config.optimizer)(
            self.model.parameters(),
            **self._config.optimizer_kwargs,
        )
        
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                **self._config.scheduler_kwargs)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self._config.monitor_metric,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        inputs, depth_mask = batch
        inputs = inputs.reshape(
            inputs.shape[0],
            3,
            self._config.data_config.crop_size[0],
            self._config.data_config.crop_size[1],
        )
        inputs = inputs.float()
        dpt_output = make_prediction_dpt(model=self.model, img=inputs)
        loss = self.criterion(dpt_output, depth_mask)
        self.log("train_loss", loss, on_step=False,on_epoch=True)
        mse = self.mse(dpt_output, depth_mask)
        self.log('train_MSE_loss', mse, on_step=False, on_epoch=True)
        for metric_name in self.metrics:
            metric = self.metrics[metric_name](dpt_output, depth_mask)
            self.log(f'train_{metric_name}', metric, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, depth_mask = batch
        inputs = inputs.float()
        dpt_output = make_prediction_dpt(self.model, inputs)
        mse = self.mse(dpt_output, depth_mask)
        loss = self.criterion(dpt_output, depth_mask)
        self.log(f"val_{self._config.data_config.dataset_names[dataloader_idx]}_{self._config.losses[0].loss_fn.split('.')[-1]}", loss, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f"val_{self._config.data_config.dataset_names[dataloader_idx]}_MSE_loss", mse, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        for metric_name in self.metrics:
            metric = self.metrics[metric_name](dpt_output, depth_mask)
            self.log(f'val_{self._config.data_config.dataset_names[dataloader_idx]}_{metric_name}', metric, on_epoch=True, add_dataloader_idx=False)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, depth_mask = batch
        inputs = inputs.reshape(
            inputs.shape[0],
            3,
            self._config.data_config.crop_size[0],
            self._config.data_config.crop_size[1],
        )
        inputs = inputs.float()

        dpt_output = make_prediction_dpt(self.model, inputs)
        mse = self.mse(dpt_output, depth_mask)
        self.log(f'MSE', mse, on_epoch=True, sync_dist=True) 
        for metric_name in self.metrics:
            metric = self.metrics[metric_name](dpt_output, depth_mask)
            self.log(f'{metric_name}', metric, on_epoch=True)

    def predict_step(self, batch):

        inputs, depth_masks = batch

        inputs = inputs.reshape(
            inputs.shape[0],
            3,
            self._config.data_config.crop_size[0],
            self._config.data_config.crop_size[1],
        )
        inputs = inputs.float()
        return make_prediction_dpt(model=self.model, img=inputs)
