from transformers import Dinov2Config, DPTForDepthEstimation, DPTConfig, Dinov2Model
import torch
import torch.nn as nn
from mmcv.cnn import build_upsample_layer


class DinoV2_DPT(nn.Module):
    """
    Модель, объединяющая DinoV2 backbone с DPT головой для оценки глубины.
    """
    def __init__(self):
        super(DinoV2_DPT, self).__init__()
        
        self.backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-large",
            out_features=["stage1", "stage2", "stage3", "stage4"],
            reshape_hidden_states=False
        )
        
        self.backbone = Dinov2Model.from_pretrained(
            "facebook/dinov2-large", 
            config=self.backbone_config
        )
        

        self.dpt_config = DPTConfig(backbone_config=self.backbone_config)
        
        self.dpt_head = DPTForDepthEstimation(config=self.dpt_config)
        
        self.dpt_head.backbone = self.backbone
        
    def forward(self, pixel_values):
        """
        Forward pass через DinoV2 -> DPT.
        
        Args:
            pixel_values (Tensor): Входные изображения [B, C, H, W]
            
        Returns:
            Tensor: Предсказанная карта глубины
        """
        outputs = self.dpt_head(pixel_values)
        
        return outputs.predicted_depth


class CustomDPT(nn.Module):
    """
    Кастомная DPT модель, которая предсказывает карту высот как вектор, 
    а затем восстанавливает из него 2D карту высот.
    """
    def __init__(self, output_height=384, output_width=384):
        super(CustomDPT, self).__init__()
        
        self.output_height = output_height
        self.output_width = output_width
        self.vector_size = output_height * output_width

        self.dpt_model = DPTForDepthEstimation.from_pretrained('Intel/dpt-large')

        self.additional_decoder_block = nn.Sequential(
            build_upsample_layer(
                {'type': "deconv"},
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        
        self.conv_2x2 = nn.Conv2d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=(2, 2), 
            padding=0
        )
        
        self.conv_1x1 = nn.Conv2d(
            in_channels=64, 
            out_channels=128, 
            kernel_size=(1, 1), 
            padding=0
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
        
        self.regression_layer = nn.Sequential(
            nn.Linear(128 * 16 * 16, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, self.vector_size) 
        )

        self.height_map_decoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, pixel_values, return_vector=False):
        """
        Forward pass модели CustomDPT.

        Args:
            pixel_values (Tensor): Входные изображения
            return_vector (bool): Если True, возвращает только вектор высот

        Returns:
            Tensor: Вектор высот (если return_vector=True) или 2D карта высот
        """

        dpt_output = self.dpt_model(pixel_values)
        predicted_depth = dpt_output.predicted_depth
        
        if len(predicted_depth.shape) == 3:
            predicted_depth = predicted_depth.unsqueeze(1)
        

        x = self.additional_decoder_block(predicted_depth)
        

        x = self.conv_2x2(x)
        x = torch.relu(x)
        
        x = self.conv_1x1(x)
        x = torch.relu(x)
        
        x = self.adaptive_pool(x)
        
        x = x.view(x.size(0), -1)
        
        height_vector = self.regression_layer(x)
        
        if return_vector:
            return height_vector
        
        batch_size = height_vector.size(0)
        height_map = height_vector.view(batch_size, 1, self.output_height, self.output_width)

        height_map = self.height_map_decoder(height_map)
        
        return height_map, height_vector


class DPT(nn.Module):
    """
    Простая обертка для предобученной DPT модели.
    """
    def __init__(self, model_name='Intel/dpt-large'):
        super(DPT, self).__init__()
        
        # Загрузка предобученной DPT модели
        self.dpt_model = DPTForDepthEstimation.from_pretrained(model_name)
    
    def forward(self, pixel_values):
        """
        Forward pass через предобученную DPT модель.
        
        Args:
            pixel_values (Tensor): Входные изображения [B, C, H, W]
            
        Returns:
            Tensor: Предсказанная карта глубины
        """
        outputs = self.dpt_model(pixel_values)
        return outputs.predicted_depth