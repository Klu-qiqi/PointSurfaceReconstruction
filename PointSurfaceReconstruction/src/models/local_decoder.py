import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class LocalDecoder(nn.Module):
    """
    Локальный декодер для предсказания SDF в окрестности точек
    
    Args:
        input_dim: Размерность входных точек
        latent_size: Размер латентного пространства
        local_feat_size: Размер локальных признаков
        hidden_dims: Список размеров скрытых слоев
        dropout: Rate dropout
        use_batch_norm: Использовать batch normalization
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        latent_size: int = 256,
        local_feat_size: int = 128,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_size = latent_size
        self.local_feat_size = local_feat_size
        self.hidden_dims = hidden_dims or [512, 512, 256]
        self.use_batch_norm = use_batch_norm
        
        # Вход: локальные признаки + координаты точки
        input_size = local_feat_size + input_dim
        
        # Построение MLP
        layers = []
        prev_dim = input_size
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU(inplace=True))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Выходной слой (SDF значение)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Инициализация весов
        self.apply(self._init_weights)
        
        logger.info(f"Initialized LocalDecoder with hidden dims {self.hidden_dims}")

    def _init_weights(self, module):
        """Инициализация весов"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self, 
        local_features: torch.Tensor, 
        points: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass локального декодера
        
        Args:
            local_features: Локальные признаки [batch_size, num_points, local_feat_size]
            points: Координаты точек [batch_size, num_points, input_dim]
            
        Returns:
            sdf_pred: Предсказанные SDF значения [batch_size, num_points]
        """
        batch_size, num_points, _ = points.shape
        
        # Конкатенируем локальные признаки с координатами точек
        x = torch.cat([local_features, points], dim=-1)  # [BS, N, local_feat_size + 3]
        
        # Reshape для MLP: [BS * N, local_feat_size + 3]
        x = x.view(-1, x.shape[-1])
        
        # Пропускаем через MLP
        sdf = self.mlp(x)  # [BS * N, 1]
        
        # Reshape обратно: [BS, N]
        sdf = sdf.view(batch_size, num_points)
        
        return sdf

    def compute_local_sdf(
        self,
        local_features: torch.Tensor,
        query_points: torch.Tensor,
        center_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисление SDF для локальных запросов относительно центров
        
        Args:
            local_features: Локальные признаки центров [BS, num_centers, local_feat_size]
            query_points: Точки запроса [BS, num_queries, 3]
            center_points: Центры локальных областей [BS, num_centers, 3]
            
        Returns:
            sdf_pred: Предсказанные SDF значения [BS, num_queries]
        """
        batch_size, num_centers, feat_dim = local_features.shape
        num_queries = query_points.shape[1]
        
        # Расширяем для всех комбинаций центров и запросов
        local_features_expanded = local_features.unsqueeze(2).repeat(1, 1, num_queries, 1)  # [BS, C, Q, F]
        center_points_expanded = center_points.unsqueeze(2).repeat(1, 1, num_queries, 1)    # [BS, C, Q, 3]
        query_points_expanded = query_points.unsqueeze(1).repeat(1, num_centers, 1, 1)      # [BS, C, Q, 3]
        
        # Относительные координаты
        relative_coords = query_points_expanded - center_points_expanded  # [BS, C, Q, 3]
        
        # Конкатенируем признаки с относительными координатами
        combined = torch.cat([
            local_features_expanded, 
            relative_coords
        ], dim=-1)  # [BS, C, Q, F + 3]
        
        # Reshape для MLP
        combined = combined.view(-1, feat_dim + 3)  # [BS * C * Q, F + 3]
        
        # Пропускаем через MLP
        sdf_all = self.mlp(combined)  # [BS * C * Q, 1]
        sdf_all = sdf_all.view(batch_size, num_centers, num_queries)  # [BS, C, Q]
        
        # Агрегируем по центрам (берем минимум по абсолютному значению)
        sdf_aggregated, _ = torch.min(sdf_all, dim=1)  # [BS, Q]
        
        return sdf_aggregated


class ResidualLocalDecoder(nn.Module):
    """
    Локальный декодер с residual connections
    """
    
    def __init__(self, input_dim: int = 131, hidden_dims: List[int] = None):
        super().__init__()
        
        self.hidden_dims = hidden_dims or [512, 512, 256, 128]
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            layers.append(ResidualBlock(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return self.output_layer(x).squeeze(-1)


class ResidualBlock(nn.Module):
    """Residual block с batch norm и dropout"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        
        # Shortcut connection
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out