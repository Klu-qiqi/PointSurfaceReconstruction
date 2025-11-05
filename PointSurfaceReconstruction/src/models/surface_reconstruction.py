import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import logging

from .pointnet import ImprovedPointNet
from .local_decoder import LocalDecoder
from .global_decoder import GlobalDecoder

logger = logging.getLogger(__name__)


class SurfaceReconstructionModel(nn.Module):
    """
    Полная модель для реконструкции поверхностей из точечных облаков
    
    Args:
        config: Конфигурация модели
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        model_config = config.get('model', {})
        
        # Инициализация компонентов
        self.pointnet = ImprovedPointNet(
            latent_size=model_config.get('latent_size', 256),
            input_dim=model_config.get('input_dim', 3),
            conv_channels=model_config.get('conv_channels', [64, 128, 256]),
            use_transform=model_config.get('use_transform', True)
        )
        
        self.local_decoder = LocalDecoder(
            input_dim=model_config.get('input_dim', 3),
            latent_size=model_config.get('latent_size', 256),
            local_feat_size=model_config.get('local_feat_size', 128),
            hidden_dims=model_config.get('decoder_hidden_dims', [512, 512, 256]),
            dropout=model_config.get('dropout_rate', 0.1),
            use_batch_norm=model_config.get('use_batch_norm', True)
        )
        
        self.global_decoder = GlobalDecoder(
            global_latent_size=model_config.get('latent_size', 256),
            local_latent_size=model_config.get('local_feat_size', 128),
            output_dim=1,
            hidden_dims=model_config.get('global_decoder_hidden_dims', [512, 512, 256, 128]),
            dropout=model_config.get('dropout_rate', 0.1),
            use_batch_norm=model_config.get('use_batch_norm', True)
        )
        
        # Weight для комбинации локальных и глобальных предсказаний
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        logger.info("Initialized SurfaceReconstructionModel")

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass полной модели
        
        Args:
            points: Тензор точек [batch_size, num_points, 3]
            
        Returns:
            sdf_pred: Предсказанные SDF значения [batch_size, num_points]
            features: Словарь с промежуточными признаками
        """
        # Извлечение признаков через PointNet
        global_feat, local_feat = self.pointnet(points)
        
        # Локальное предсказание SDF
        local_sdf = self.local_decoder(local_feat, points)
        
        # Глобальное предсказание SDF
        global_sdf = self.global_decoder(global_feat, local_feat, points)
        
        # Комбинирование предсказаний
        sdf_pred = self.alpha * local_sdf + (1 - self.alpha) * global_sdf
        
        # Сохраняем промежуточные результаты для анализа
        features = {
            'global_features': global_feat,
            'local_features': local_feat,
            'local_sdf': local_sdf,
            'global_sdf': global_sdf,
            'alpha': self.alpha
        }
        
        return sdf_pred, features

    def compute_gradients(
        self, 
        points: torch.Tensor, 
        create_graph: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычисление SDF и градиентов
        
        Args:
            points: Точки для вычисления [batch_size, num_points, 3]
            create_graph: Сохранять граф для высших производных
            
        Returns:
            sdf: SDF значения [batch_size, num_points]
            gradients: Градиенты SDF [batch_size, num_points, 3]
        """
        points.requires_grad_(True)
        
        sdf, _ = self.forward(points)
        
        # Вычисление градиентов
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=create_graph,
            retain_graph=create_graph,
            only_inputs=True
        )[0]
        
        return sdf, gradients

    def project_to_surface(
        self, 
        points: torch.Tensor, 
        num_iterations: int = 3
    ) -> torch.Tensor:
        """
        Проекция точек на поверхность с использованием SDF
        
        Args:
            points: Исходные точки [batch_size, num_points, 3]
            num_iterations: Количество итераций проекции
            
        Returns:
            surface_points: Точки на поверхности [batch_size, num_points, 3]
        """
        current_points = points.clone()
        
        for i in range(num_iterations):
            current_points.requires_grad_(True)
            
            # Вычисляем SDF и градиенты
            sdf, gradients = self.compute_gradients(current_points)
            
            # Нормализуем градиенты
            grad_norm = torch.nn.functional.normalize(gradients, dim=-1)
            
            # Проекция: p_surface = p - SDF(p) * ∇SDF/‖∇SDF‖
            current_points = current_points - sdf.unsqueeze(-1) * grad_norm
            
            # Detach для экономии памяти
            if i < num_iterations - 1:
                current_points = current_points.detach()
        
        return current_points

    def get_parameters(self) -> Dict[str, nn.Parameter]:
        """Получение параметров всех компонентов"""
        return {
            'pointnet': list(self.pointnet.parameters()),
            'local_decoder': list(self.local_decoder.parameters()),
            'global_decoder': list(self.global_decoder.parameters())
        }

    def save_model(self, path: str):
        """Сохранение модели"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str, device: torch.device):
        """Загрузка модели"""
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")