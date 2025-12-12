"""
Конфигурация экспериментов
"""
import torch

# Общие параметры
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Параметры denoising автоэнкодера
DENOISING_CONFIG = {
    'batch_size': 256,
    'epochs': 100,
    'lr': 1e-3,
    'noise_factor': 0.5,
    'weight_decay': 1e-5,
}

# Параметры данных
DATA_CONFIG = {
    'image_size': 28,
    'num_workers': 2,
    'pin_memory': True,
}
