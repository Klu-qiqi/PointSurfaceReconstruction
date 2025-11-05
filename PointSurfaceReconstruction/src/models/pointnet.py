import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImprovedPointNet(nn.Module):
    """
    Улучшенная реализация PointNet для извлечения признаков из точечных облаков
    
    Args:
        latent_size: Размер латентного пространства
        input_dim: Размерность входных точек (по умолчанию 3)
        conv_channels: Каналы сверточных слоев
        use_transform: Использовать spatial transformer
        feature_transform: Использовать feature transformer
    """
    
    def __init__(
        self,
        latent_size: int = 256,
        input_dim: int = 3,
        conv_channels: list = None,
        use_transform: bool = True,
        feature_transform: bool = False
    ):
        super().__init__()
        
        self.latent_size = latent_size
        self.input_dim = input_dim
        self.conv_channels = conv_channels or [64, 128, 256]
        self.use_transform = use_transform
        self.feature_transform = feature_transform
        
        # Spatial transformer (T-Net)
        if self.use_transform:
            self.transform = self._build_transform_net()
        
        # Feature transformer
        if self.feature_transform:
            self.feature_transform_net = self._build_feature_transform_net(64)
        
        # Основные сверточные слои
        conv_layers = []
        in_channels = input_dim
        
        for out_channels in self.conv_channels:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        
        # Финальный слой для латентного представления
        conv_layers.extend([
            nn.Conv1d(in_channels, latent_size, 1),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(inplace=True)
        ])
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Локальные и глобальные преобразования
        self.local_transform = nn.Sequential(
            nn.Linear(self.conv_channels[0], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )
        
        self.global_transform = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, latent_size)
        )
        
        logger.info(f"Initialized PointNet with latent size {latent_size}")

    def _build_transform_net(self) -> nn.Module:
        """Построение spatial transformer network"""
        return nn.Sequential(
            nn.Conv1d(self.input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.input_dim * self.input_dim)
        )

    def _build_feature_transform_net(self, feature_dim: int) -> nn.Module:
        """Построение feature transformer network"""
        return nn.Sequential(
            nn.Conv1d(feature_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim * feature_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass PointNet
        
        Args:
            x: Тензор точек [batch_size, num_points, input_dim]
            
        Returns:
            global_features: Глобальные признаки [batch_size, latent_size]
            local_features: Локальные признаки [batch_size, num_points, local_feat_size]
        """
        batch_size, num_points, input_dim = x.shape
        
        # Применяем spatial transform если нужно
        if self.use_transform and input_dim == 3:
            transform_matrix = self.transform(x.transpose(1, 2))
            transform_matrix = transform_matrix.view(-1, input_dim, input_dim)
            x_transformed = torch.bmm(x, transform_matrix)
            x = x_transformed
        
        # Подготовка для Conv1d: [BS, N, C] -> [BS, C, N]
        x = x.transpose(1, 2)
        
        # Пропускаем через сверточные слои
        features = []
        current_x = x
        
        for i, layer in enumerate(self.conv_layers):
            current_x = layer(current_x)
            if i in [2, 5]:  # Сохраняем промежуточные features
                features.append(current_x)
        
        # Локальные признаки (после первого блока)
        local_feat = features[0]  # [BS, 64, N]
        
        # Применяем feature transform если нужно
        if self.feature_transform and len(features) > 0:
            feature_transform = self.feature_transform_net(local_feat)
            feature_transform = feature_transform.view(-1, 64, 64)
            local_feat = torch.bmm(local_feat.transpose(1, 2), feature_transform).transpose(1, 2)
        
        # Глобальные признаки через max pooling
        global_feat = torch.max(current_x, 2)[0]  # [BS, latent_size]
        global_feat = self.global_transform(global_feat)
        
        # Подготовка локальных признаков
        local_feat = local_feat.transpose(1, 2)  # [BS, N, 64]
        local_feat = self.local_transform(local_feat.view(-1, 64)).view(batch_size, num_points, -1)
        
        return global_feat, local_feat

    def get_parameters(self):
        """Получение параметров для оптимизатора"""
        return self.parameters()


class PointNetWithProj(nn.Module):
    """
    PointNet с дополнительным проекционным модулем для SDF
    """
    
    def __init__(self, latent_size: int = 256, input_dim: int = 3):
        super().__init__()
        self.pointnet = ImprovedPointNet(latent_size, input_dim)
        
        # Проекционный модуль для SDF
        self.sdf_projection = nn.Sequential(
            nn.Linear(latent_size + input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Предсказание SDF напрямую"""
        global_feat, local_feat = self.pointnet(points)
        
        # Повторяем глобальные признаки для каждой точки
        batch_size, num_points, _ = points.shape
        global_feat_expanded = global_feat.unsqueeze(1).repeat(1, num_points, 1)
        
        # Конкатенируем с координатами точек
        combined = torch.cat([global_feat_expanded, points], dim=-1)
        
        # Проецируем в SDF
        sdf = self.sdf_projection(combined).squeeze(-1)
        
        return sdf