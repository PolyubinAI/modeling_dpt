# modeling-dpt

 
## Быстрый запуск:
- ```make install``` - установка зависимостей (см. requirements.txt)
- ```make train``` - старт обучения

## Настройка обучения и гиперпарметры:
- [Файл конфигурации](./configs/config.yaml)

 ## Пример использования

```python
import torch
from transformers import DPTForDepthEstimation
from predict_utils import make_prediction_dpt

dummy_tensor = torch.empty((1, 3, 512, 512), dtype=torch.float32)

model_path = <YOUR_DATA_PATH>


model = DPTForDepthEstimation.from_pretrained(
            pretrained_model_name_or_path=model_path,
        )
preds = make_prediction_dpt(model, dummy_tensor)

```
