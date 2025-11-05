import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SurfaceReconstructionLoss(nn.Module):
    """
    Комплексная функция потерь для реконструкции поверхностей
    """
    
    def __init__(self, loss_weights: Optional[Dict] = None):
        super().__init__()
        
        self.loss_weights = loss_weights or {
            'sdf_loss': 1.0,
            'surface_loss': 0.1,
            'normal_loss': 0.01,
            'eikonal_loss': 0.1,
            'curvature_loss': 0.001
        }
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        logger.info(f"Initialized SurfaceReconstructionLoss with weights: {self.loss_weights}")

    def forward(
        self,
        sdf_pred: torch.Tensor,
        sdf_gt: torch.Tensor,
        points: torch.Tensor,
        normals_pred: Optional[torch.Tensor] = None,
        normals_gt: Optional[torch.Tensor] = None,
        gradients_pred: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Вычисление общей функции потерь
        
        Args:
            sdf_pred: Предсказанные SDF значения [BS, N]
            sdf_gt: GT SDF значения [BS, N]
            points: Точки [BS, N, 3]
            normals_pred: Предсказанные нормали [BS, N, 3]
            normals_gt: GT нормали [BS, N, 3]
            gradients_pred: Предсказанные градиенты [BS, N, 3]
            
        Returns:
            total_loss: Общий loss
            loss_dict: Словарь с компонентами loss
        """
        loss_dict = {}
        
        # Основной SDF loss
        loss_dict['sdf_loss'] = self.mse_loss(sdf_pred, sdf_gt)
        
        # Surface consistency loss
        loss_dict['surface_loss'] = self._compute_surface_consistency_loss(sdf_pred, sdf_gt)
        
        # Normal consistency loss если есть нормали
        if normals_pred is not None and normals_gt is not None:
            loss_dict['normal_loss'] = self._compute_normal_consistency_loss(normals_pred, normals_gt)
        
        # Eikonal regularization
        if gradients_pred is not None:
            loss_dict['eikonal_loss'] = self._compute_eikonal_loss(gradients_pred)
        else:
            # Вычисляем градиенты если не предоставлены
            gradients = self._compute_gradients(sdf_pred, points)
            loss_dict['eikonal_loss'] = self._compute_eikonal_loss(gradients)
        
        # Curvature regularization
        loss_dict['curvature_loss'] = self._compute_curvature_regularization(sdf_pred, points)
        
        # Вычисление общего loss с весами
        total_loss = 0.0
        for loss_name, loss_value in loss_dict.items():
            weight = self.loss_weights.get(loss_name, 1.0)
            total_loss += weight * loss_value
        
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict

    def _compute_surface_consistency_loss(
        self,
        sdf_pred: torch.Tensor,
        sdf_gt: torch.Tensor
    ) -> torch.Tensor:
        """Loss для консистентности поверхности"""
        # Для точек около поверхности предсказания должны быть точными
        surface_mask = torch.abs(sdf_gt) < 0.01
        if torch.any(surface_mask):
            return self.l1_loss(sdf_pred[surface_mask], sdf_gt[surface_mask])
        return torch.tensor(0.0, device=sdf_pred.device)

    def _compute_normal_consistency_loss(
        self,
        normals_pred: torch.Tensor,
        normals_gt: torch.Tensor
    ) -> torch.Tensor:
        """Loss для консистентности нормалей"""
        cos_similarity = F.cosine_similarity(normals_pred, normals_gt, dim=-1)
        # Максимизируем cosine similarity (минимизируем 1 - cos_sim)
        normal_loss = 1 - torch.mean(cos_similarity)
        return normal_loss

    def _compute_eikonal_loss(self, gradients: torch.Tensor) -> torch.Tensor:
        """Eikonal regularization loss"""
        grad_norms = torch.norm(gradients, dim=-1)
        eikonal_loss = torch.mean((grad_norms - 1) ** 2)
        return eikonal_loss

    def _compute_curvature_regularization(
        self,
        sdf: torch.Tensor,
        points: torch.Tensor
    ) -> torch.Tensor:
        """Curvature regularization через вторые производные"""
        batch_size, num_points = sdf.shape
        
        # Вычисляем градиенты с созданием графа
        points.requires_grad_(True)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Вычисляем divergence градиентов (лапласиан)
        curvature_loss = 0.0
        for dim in range(3):
            grad_component = gradients[:, :, dim]
            
            second_grad = torch.autograd.grad(
                outputs=grad_component,
                inputs=points,
                grad_outputs=torch.ones_like(grad_component),
                create_graph=False,
                retain_graph=True
            )[0]
            
            if second_grad is not None:
                # Берем только диагональные элементы гессиана
                curvature_loss += torch.mean(second_grad[:, :, dim] ** 2)
        
        return curvature_loss / 3.0

    def _compute_gradients(
        self,
        sdf: torch.Tensor,
        points: torch.Tensor
    ) -> torch.Tensor:
        """Вычисление градиентов SDF"""
        points.requires_grad_(True)
        
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True
        )[0]
        
        return gradients


class AdaptiveLossWeightScheduler:
    """
    Адаптивный scheduler для весов loss функций
    """
    
    def __init__(self, initial_weights: Dict, update_frequency: int = 100):
        self.initial_weights = initial_weights.copy()
        self.current_weights = initial_weights.copy()
        self.update_frequency = update_frequency
        self.loss_history = {name: [] for name in initial_weights.keys()}
        
    def update_weights(self, current_losses: Dict[str, float], iteration: int) -> Dict[str, float]:
        """
        Обновление весов на основе истории loss'ов
        
        Args:
            current_losses: Текущие значения loss'ов
            iteration: Текущая итерация
            
        Returns:
            Обновленные веса
        """
        if iteration % self.update_frequency != 0:
            return self.current_weights
        
        # Обновляем историю
        for loss_name, loss_value in current_losses.items():
            if loss_name in self.loss_history:
                self.loss_history[loss_name].append(loss_value)
                
                # Держим только последние N значений
                if len(self.loss_history[loss_name]) > 100:
                    self.loss_history[loss_name].pop(0)
        
        # Вычисляем новые веса на основе относительных величин loss'ов
        avg_losses = {}
        for loss_name, history in self.loss_history.items():
            if history:
                avg_losses[loss_name] = np.mean(history)
        
        if len(avg_losses) > 1:
            # Нормализуем loss'ы
            min_loss = min(avg_losses.values())
            max_loss = max(avg_losses.values())
            
            if max_loss > min_loss:
                for loss_name, avg_loss in avg_losses.items():
                    # Вес обратно пропорционален относительной величине loss'а
                    normalized_loss = (avg_loss - min_loss) / (max_loss - min_loss)
                    new_weight = self.initial_weights[loss_name] * (1.0 - normalized_loss * 0.5)
                    self.current_weights[loss_name] = max(new_weight, 0.1)
        
        return self.current_weights