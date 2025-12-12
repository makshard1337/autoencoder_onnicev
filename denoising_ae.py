"""
Модуль denoising автоэнкодера
Реализация автоэнкодера для восстановления зашумленных изображений
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from config import SEED, DEVICE, DENOISING_CONFIG
from utils import set_seed, count_parameters, add_noise, calculate_psnr, calculate_ssim

# Фиксируем seed для воспроизводимости
set_seed(SEED)

class FCDenoisingAE(nn.Module):
    """
    Полносвязный denoising автоэнкодер
    Использует только полносвязные слои (без сверток)
    """
    def __init__(self, input_size=28, hidden_dims=[512, 256, 128]):
        super(FCDenoisingAE, self).__init__()
        self.input_size = input_size
        input_dim = input_size * input_size  # 28*28 = 784
        
        # Энкодер: сжимает изображение до латентного представления
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.latent_dim = hidden_dims[-1]  # Размерность латентного пространства
        
        # Декодер: восстанавливает изображение из латентного представления
        # Симметричная структура (обратный порядок размерностей)
        decoder_layers = []
        hidden_dims_rev = hidden_dims[::-1]  # Обращаем порядок
        prev_dim = self.latent_dim
        for hidden_dim in hidden_dims_rev[1:]:  # Пропускаем первый (он уже есть)
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Финальный слой восстанавливает исходный размер
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # Нормализация к [0, 1]
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        # Преобразуем изображение в вектор
        x_flat = x.view(x.size(0), -1)
        # Кодируем в латентное представление
        encoded = self.encoder(x_flat)
        # Декодируем обратно в изображение
        decoded = self.decoder(encoded)
        # Преобразуем обратно в изображение 28x28
        return decoded.view(x.size(0), 1, self.input_size, self.input_size)

class ConvDenoisingAE(nn.Module):
    """Сверточный denoising автоэнкодер"""
    def __init__(self, input_size=28, latent_dim=128):
        super(ConvDenoisingAE, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Энкодер
        self.encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 7x7 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Переход к латентному представлению
        self.fc_encoder = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Переход от латентного представления
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 4 * 4)
        )
        
        # Декодер (точные размеры для восстановления 28x28)
        self.decoder = nn.Sequential(
            # 4x4 -> 7x7
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 7x7 -> 14x14
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Энкодирование
        encoded = self.encoder(x)
        encoded_flat = encoded.view(encoded.size(0), -1)
        latent = self.fc_encoder(encoded_flat)
        
        # Декодирование
        decoded_flat = self.fc_decoder(latent)
        decoded_flat = decoded_flat.view(decoded_flat.size(0), 128, 4, 4)
        reconstructed = self.decoder(decoded_flat)
        
        # Обрезаем до правильного размера (28x28), если получилось больше
        # Это может произойти из-за округлений в ConvTranspose2d
        if reconstructed.shape[2] > self.input_size or reconstructed.shape[3] > self.input_size:
            reconstructed = reconstructed[:, :, :self.input_size, :self.input_size]
        # Или дополняем, если получилось меньше
        elif reconstructed.shape[2] < self.input_size or reconstructed.shape[3] < self.input_size:
            pad_h = self.input_size - reconstructed.shape[2]
            pad_w = self.input_size - reconstructed.shape[3]
            reconstructed = F.pad(reconstructed, (0, pad_w, 0, pad_h), mode='reflect')
        
        return reconstructed

def train_denoising_ae(model, train_loader, test_loader=None, epochs=50, 
                       lr=1e-3, noise_factor=0.5, weight_decay=1e-5, device=DEVICE):
    """
    Обучает denoising автоэнкодер
    
    Процесс:
    1. Берем чистое изображение
    2. Добавляем к нему шум
    3. Подаем зашумленное изображение в модель
    4. Модель пытается восстановить чистое изображение
    5. Сравниваем восстановленное с оригиналом (MSE loss)
    
    Returns:
        history: словарь с историей обучения (потери, метрики)
    """
    model = model.to(device)
    # Используем MSE loss для сравнения восстановленного и оригинального изображения
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_psnr': [],
        'test_ssim': [],
        'epoch': []
    }
    
    print(f"Модель имеет {count_parameters(model):,} параметров")
    print(f"Начинаем обучение на {device}...")
    print(f"Количество эпох: {epochs}, Learning rate: {lr}, Noise factor: {noise_factor}")
    
    for epoch in range(epochs):
        # === ОБУЧЕНИЕ ===
        model.train()  # Режим обучения
        train_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, _ in pbar:
            images = images.to(device)
            
            # Добавляем гауссов шум к изображениям
            noisy_images = add_noise(images, noise_factor)
            
            # Модель восстанавливает изображение из зашумленного
            reconstructed = model(noisy_images)
            
            # Вычисляем потерю (сравниваем с оригиналом)
            loss = criterion(reconstructed, images)
            
            # Обновление весов
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / num_batches
        history['train_loss'].append(avg_train_loss)
        
        # === ВАЛИДАЦИЯ (если есть тестовая выборка) ===
        if test_loader is not None:
            model.eval()  # Режим оценки (без обновления весов)
            test_loss = 0
            test_psnr = 0
            test_ssim = 0
            num_test_batches = 0
            
            with torch.no_grad():  # Не вычисляем градиенты для ускорения
                for images, _ in test_loader:
                    images = images.to(device)
                    noisy_images = add_noise(images, noise_factor)
                    reconstructed = model(noisy_images)
                    
                    loss = criterion(reconstructed, images)
                    test_loss += loss.item()
                    
                    # Вычисляем метрики качества для каждого изображения
                    for i in range(images.size(0)):
                        test_psnr += calculate_psnr(images[i], reconstructed[i])
                        test_ssim += calculate_ssim(images[i], reconstructed[i])
                    
                    num_test_batches += 1
            
            # Средние значения метрик
            avg_test_loss = test_loss / num_test_batches
            avg_psnr = test_psnr / (num_test_batches * test_loader.batch_size)
            avg_ssim = test_ssim / (num_test_batches * test_loader.batch_size)
            
            history['test_loss'].append(avg_test_loss)
            history['test_psnr'].append(avg_psnr)
            history['test_ssim'].append(avg_ssim)
            
            # Выводим информацию каждые 5 эпох
            if (epoch + 1) % 5 == 0:
                print(f'\nЭпоха {epoch+1}/{epochs}:')
                print(f'  Потеря на обучении: {avg_train_loss:.4f}')
                print(f'  Потеря на тесте: {avg_test_loss:.4f}')
                print(f'  PSNR: {avg_psnr:.2f} dB (чем выше, тем лучше)')
                print(f'  SSIM: {avg_ssim:.4f} (чем ближе к 1, тем лучше)')
        else:
            if (epoch + 1) % 5 == 0:
                print(f'Эпоха {epoch+1}/{epochs}, Потеря на обучении: {avg_train_loss:.4f}')
        
        history['epoch'].append(epoch + 1)
    
    print("\nОбучение завершено!")
    return history

def evaluate_denoising_ae(model, test_loader, noise_factor=0.5, device=DEVICE):
    """
    Оценивает качество denoising автоэнкодера
    
    Returns:
        metrics: словарь с метриками
    """
    model.eval()
    model = model.to(device)
    
    criterion = nn.MSELoss()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_samples = 0
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc='Оценка модели'):
            images = images.to(device)
            noisy_images = add_noise(images, noise_factor)
            reconstructed = model(noisy_images)
            
            loss = criterion(reconstructed, images)
            total_loss += loss.item() * images.size(0)
            
            for i in range(images.size(0)):
                total_psnr += calculate_psnr(images[i], reconstructed[i])
                total_ssim += calculate_ssim(images[i], reconstructed[i])
            
            num_samples += images.size(0)
    
    metrics = {
        'mse': total_loss / num_samples,
        'psnr': total_psnr / num_samples,
        'ssim': total_ssim / num_samples
    }
    
    print(f'MSE: {metrics["mse"]:.6f}')
    print(f'PSNR: {metrics["psnr"]:.2f} dB')
    print(f'SSIM: {metrics["ssim"]:.4f}')
    
    return metrics
