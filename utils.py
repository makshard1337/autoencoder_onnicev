"""
Вспомогательные функции для загрузки данных и визуализации
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import seaborn as sns
from config import SEED, DATA_CONFIG

# Фиксация seed для воспроизводимости
def set_seed(seed=SEED):
    """Фиксирует случайные зерна во всех библиотеках"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_mnist_loader(batch_size=256, train=True, num_samples_per_class=None, 
                     num_classes=10, use_augmentation=False):
    """
    Загружает MNIST датасет с опциональной выборкой подмножества классов
    
    Args:
        batch_size: размер батча
        train: загружать обучающую или тестовую выборку
        num_samples_per_class: количество образцов на класс (для контрастивного обучения)
        num_classes: количество классов для использования
        use_augmentation: использовать ли аугментации
    """
    set_seed()
    
    transform_list = [transforms.ToTensor()]
    
    if use_augmentation:
        transform_list = [
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1),  # Gaussian noise
        ]
    transform = transforms.Compose(transform_list)
    dataset = datasets.MNIST(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )
    # Выборка подмножества классов и образцов
    labels = dataset.targets.numpy()
    selected_classes = list(range(min(num_classes, 10)))
    if train and num_samples_per_class is not None:
        # Для обучающей выборки: ограничиваем количество образцов на класс
        indices = []
        for cls in selected_classes:
            cls_indices = np.where(labels == cls)[0]
            if len(cls_indices) > num_samples_per_class:
                np.random.seed(SEED)
                cls_indices = np.random.choice(cls_indices, num_samples_per_class, replace=False)
            indices.extend(cls_indices)
        dataset = Subset(dataset, indices)
    elif not train:
        # Для тестовой выборки: ограничиваем количество образцов на класс для справедливого сравнения
        # Используем то же количество или немного больше (например, 2x)
        test_samples_per_class = num_samples_per_class * 2 if num_samples_per_class is not None else None
        indices = []
        for cls in selected_classes:
            cls_indices = np.where(labels == cls)[0]
            if test_samples_per_class is not None and len(cls_indices) > test_samples_per_class:
                np.random.seed(SEED + 1)  # Другой seed для тестовой выборки
                cls_indices = np.random.choice(cls_indices, test_samples_per_class, replace=False)
            indices.extend(cls_indices)
        dataset = Subset(dataset, indices)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=DATA_CONFIG['num_workers'],
        pin_memory=DATA_CONFIG['pin_memory']
    )
    
    return loader

def get_fashion_mnist_loader(batch_size=256, train=True, num_samples_per_class=None,
                             num_classes=5, use_augmentation=False):
    """Загружает Fashion-MNIST датасет"""
    set_seed()
    
    transform_list = [transforms.ToTensor()]
    
    if use_augmentation:
        transform_list = [
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1),
        ]
    
    transform = transforms.Compose(transform_list)
    
    dataset = datasets.FashionMNIST(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )
    
    if num_samples_per_class is not None and train:
        indices = []
        labels = dataset.targets.numpy()
        selected_classes = list(range(min(num_classes, 10)))
        
        for cls in selected_classes:
            cls_indices = np.where(labels == cls)[0]
            if len(cls_indices) > num_samples_per_class:
                np.random.seed(SEED)
                cls_indices = np.random.choice(cls_indices, num_samples_per_class, replace=False)
            indices.extend(cls_indices)
        
        dataset = Subset(dataset, indices)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=DATA_CONFIG['num_workers'],
        pin_memory=DATA_CONFIG['pin_memory']
    )
    
    return loader

def add_noise(images, noise_factor=0.5):
    """
    Добавляет гауссов шум к изображениям
    Используется для обучения denoising автоэнкодера
    
    Args:
        images: тензор изображений
        noise_factor: сила шума (стандартное отклонение)
    """
    # Генерируем случайный шум с нормальным распределением
    noise = torch.randn_like(images) * noise_factor
    # Добавляем шум и ограничиваем значения диапазоном [0, 1]
    noisy_images = torch.clamp(images + noise, 0.0, 1.0)
    return noisy_images

def visualize_denoising_results(model, test_loader, device, num_samples=8, noise_factor=0.5):
    """Визуализирует результаты denoising"""
    model.eval()
    with torch.no_grad():
        # Выбираем примеры из разных классов (не только нули)
        selected_images = []
        seen_labels = set()
        
        data_iter = iter(test_loader)
        for _ in range(20):  # Просматриваем до 20 батчей
            try:
                batch_images, batch_labels = next(data_iter)
                batch_images = batch_images.to(device)
                
                for i in range(len(batch_images)):
                    label = batch_labels[i].item()
                    # Берем по одному примеру из каждого класса
                    if label not in seen_labels:
                        selected_images.append(batch_images[i:i+1])
                        seen_labels.add(label)
                        if len(selected_images) >= num_samples:
                            break
                
                if len(selected_images) >= num_samples:
                    break
            except StopIteration:
                break
        
        # Если не набрали достаточно, добавляем любые примеры
        if len(selected_images) < num_samples:
            data_iter = iter(test_loader)
            batch_images, _ = next(data_iter)
            batch_images = batch_images.to(device)
            for i in range(len(selected_images), num_samples):
                if i < len(batch_images):
                    selected_images.append(batch_images[i:i+1])
        
        images = torch.cat(selected_images[:num_samples], dim=0)
        
        noisy_images = add_noise(images, noise_factor)
        reconstructed = model(noisy_images)
        
        # Диагностика: проверяем диапазоны значений
        print(f"Диапазон оригинальных изображений: [{images.min().item():.4f}, {images.max().item():.4f}]")
        print(f"Диапазон зашумленных изображений: [{noisy_images.min().item():.4f}, {noisy_images.max().item():.4f}]")
        print(f"Диапазон восстановленных изображений: [{reconstructed.min().item():.4f}, {reconstructed.max().item():.4f}]")
        print(f"Среднее значение восстановленных: {reconstructed.mean().item():.4f}")
        
        # Преобразуем в numpy для визуализации (убираем канальный размер если он есть)
        images_np = images.cpu().squeeze().numpy()
        noisy_np = noisy_images.cpu().squeeze().numpy()
        reconstructed_np = reconstructed.cpu().squeeze().numpy()
        
        # Если одно изображение, добавляем размерность батча
        if len(images_np.shape) == 2:
            images_np = images_np[np.newaxis, :, :]
        if len(noisy_np.shape) == 2:
            noisy_np = noisy_np[np.newaxis, :, :]
        if len(reconstructed_np.shape) == 2:
            reconstructed_np = reconstructed_np[np.newaxis, :, :]
        
        fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))
        
        for i in range(num_samples):
            # Оригинал
            img_orig = images_np[i]
            axes[0, i].imshow(img_orig, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title('Оригинал')
            axes[0, i].axis('off')
            
            # Зашумленное
            img_noisy = noisy_np[i]
            axes[1, i].imshow(img_noisy, cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title('Зашумленное')
            axes[1, i].axis('off')
            
            # Восстановленное
            img_recon = reconstructed_np[i]
            # Нормализуем для визуализации, если значения выходят за [0, 1]
            img_recon_clipped = np.clip(img_recon, 0, 1)
            axes[2, i].imshow(img_recon_clipped, cmap='gray', vmin=0, vmax=1)
            axes[2, i].set_title('Восстановленное')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        return fig

def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_psnr(img1, img2):
    """Вычисляет PSNR между двумя изображениями"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2):
    """Упрощенная версия SSIM"""
    # Упрощенная версия для демонстрации
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    sigma1_sq = torch.var(img1)
    sigma2_sq = torch.var(img2)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
    
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim.item()

