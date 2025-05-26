import numpy as np
from matplotlib import pyplot as plt
import importlib
import typing as tp
import torch


def plot_gradients(model):
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            gradients[name] = parameter.grad.cpu().detach().numpy()

    # Plot and save gradient distribution for each layer
    for name, grad in gradients.items():  # noqa: WPS440
        plt.hist(np.reshape(grad, [-1]))
        # plt.title(f"{name} Gradient Distribution")
        # plt.xlabel("Gradient Bins")
        # plt.ylabel("Frequency")
        plt.show()
        plt.close()


def calc_batch_loss(preds, gt, loss_func):
    lss = float(0)
    bs = preds.shape[0]

    prediction = torch.reshape(preds, (bs, -1))
    target = torch.reshape(gt, (bs, -1))

    for i in range(preds.shape[0]):  # noqa: WPS519
        lss += loss_func(prediction[i], target[i]).item()

    return lss


def load_object(obj_path: str, default_obj_path: str = '') -> tp.Any:
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):  # noqa: WPS421
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    return getattr(module_obj, obj_name)
