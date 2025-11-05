import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class GlobalDecoder(nn.Module):
    """
    Глобальный декодер для предсказания SDF с учетом глобальной структуры
    
    Args:
        global_latent_size: Размер глобального латентного пространства
        local_latent_size: Размер локального латентного пространства  
        output_dim: Размерность выхода (1 для SDF)
        hidden_dims: Список размеров скрытых слоев
        dropout: Rate dropout
        use_batch_norm: Использовать batch normalization
    """
    
    def __init__(
        self,
        global_latent_size: int = 256,
        local_latent_size: int = 128,
        output_dim: int = 1,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.global_latent_size = global_latent_size
        self.local_latent_size = local_latent_size
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [512, 512, 256, 128]
        self.use_batch_norm = use_batch_norm
        
        # Вход: глобальные признаки + локальные признаки + координаты точек
        input_size = global_latent_size + local_latent_size + 3
        
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
        
        # Выходной слой
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Инициализация весов
        self.apply(self._init_weights)
        
        logger.info(f"Initialized GlobalDecoder with hidden dims {self.hidden_dims}")

    def _init_weights(self, module):
        """Инициализация весов"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        points: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass глобального декодера
        
        Args:
            global_features: Глобальные признаки [batch_size, global_latent_size]
            local_features: Локальные признаки [batch_size, num_points, local_latent_size]
            points: Координаты точек [batch_size, num_points, 3]
            
        Returns:
            sdf_pred: Предсказанные SDF значения [batch_size, num_points]
        """
        batch_size, num_points, _ = points.shape
        
        # Расширяем глобальные признаки для каждой точки
        global_expanded = global_features.unsqueeze(1).repeat(1, num_points, 1)  # [BS, N, global_latent_size]
        
        # Конкатенируем все признаки
        x = torch.cat([global_expanded, local_features, points], dim=-1)  # [BS, N, global + local + 3]
        
        # Reshape для MLP
        x = x.view(-1, x.shape[-1])  # [BS * N, input_size]
        
        # Пропускаем через MLP
        sdf = self.mlp(x)  # [BS * N, 1]
        
        # Reshape обратно
        sdf = sdf.view(batch_size, num_points)
        
        return sdf


class MultiScaleGlobalDecoder(nn.Module):
    """
    Многомасштабный глобальный декодер с skip connections
    """
    
    def __init__(
        self,
        global_latent_size: int = 256,
        local_latent_sizes: List[int] = None,
        hidden_dims: List[int] = None
    ):
        super().__init__()
        
        self.global_latent_size = global_latent_size
        self.local_latent_sizes = local_latent_sizes or [128, 64, 32]
        self.hidden_dims = hidden_dims or [512, 512, 256, 128]
        
        # Многомасштабные блоки
        self.blocks = nn.ModuleList()
        
        input_size = global_latent_size + self.local_latent_sizes[0] + 3
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            block = MultiScaleBlock(
                input_size,
                hidden_dim,
                local_latent_size=self.local_latent_sizes[min(i, len(self.local_latent_sizes)-1)] if i > 0 else 0
            )
            self.blocks.append(block)
            input_size = hidden_dim
        
        self.output_layer = nn.Linear(input_size, 1)
        
    def forward(
        self,
        global_features: torch.Tensor,
        multi_scale_features: List[torch.Tensor],
        points: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_points, _ = points.shape
        
        # Начальные признаки
        global_expanded = global_features.unsqueeze(1).repeat(1, num_points, 1)
        x = torch.cat([global_expanded, multi_scale_features[0], points], dim=-1)
        x = x.view(-1, x.shape[-1])
        
        # Пропускаем через блоки
        for i, block in enumerate(self.blocks):
            local_feat = None
            if i < len(multi_scale_features) - 1:
                local_feat = multi_scale_features[i + 1].view(-1, multi_scale_features[i + 1].shape[-1])
            
            x = block(x, local_feat)
        
        # Выход
        sdf = self.output_layer(x).view(batch_size, num_points)
        return sdf


class MultiScaleBlock(nn.Module):
    """Блок с skip connections для многомасштабных признаков"""
    
    def __init__(self, input_dim: int, output_dim: int, local_latent_size: int = 0):
        super().__init__()
        
        self.local_latent_size = local_latent_size
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        
        if local_latent_size > 0:
            self.local_proj = nn.Linear(local_latent_size, output_dim)
        
    def forward(self, x: torch.Tensor, local_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        
        if local_feat is not None and self.local_latent_size > 0:
            local_proj = self.local_proj(local_feat)
            x = x + local_proj
        
        x = self.relu(x)
        return x