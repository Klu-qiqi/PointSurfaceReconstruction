import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SafeGradientCalculator:
    """
    Безопасный расчет градиентов с контролем памяти
    """
    
    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size
    
    def compute_sdf_gradient(
        self,
        sdf: torch.Tensor,
        points: torch.Tensor,
        create_graph: bool = False,
        retain_graph: bool = False
    ) -> torch.Tensor:
        """
        Безопасный расчет градиента SDF с chunking'ом
        
        Args:
            sdf: Тензор SDF значений [BS, N]
            points: Тензор точек [BS, N, 3]
            create_graph: Сохранять граф для вторых производных
            retain_graph: Сохранять граф вычислений
            
        Returns:
            Градиенты [BS, N, 3]
        """
        batch_size, num_points = sdf.shape
        
        # Если точек немного, вычисляем сразу
        if num_points <= self.chunk_size:
            return self._compute_gradients_directly(
                sdf, points, create_graph, retain_graph
            )
        
        # Иначе используем chunking
        gradients = []
        
        for i in range(0, num_points, self.chunk_size):
            end_idx = min(i + self.chunk_size, num_points)
            
            sdf_chunk = sdf[:, i:end_idx]
            points_chunk = points[:, i:end_idx].clone().requires_grad_(True)
            
            grad_chunk = self._compute_gradients_directly(
                sdf_chunk, points_chunk, create_graph, retain_graph
            )
            
            gradients.append(grad_chunk)
        
        return torch.cat(gradients, dim=1)
    
    def _compute_gradients_directly(
        self,
        sdf: torch.Tensor,
        points: torch.Tensor,
        create_graph: bool,
        retain_graph: bool
    ) -> torch.Tensor:
        """Прямой расчет градиентов"""
        grad = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=True
        )[0]
        
        if grad is None:
            # Если градиенты не требуются, возвращаем zeros
            grad = torch.zeros_like(points)
        
        return grad
    
    def compute_second_order_gradients(
        self,
        sdf: torch.Tensor,
        points: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисление градиентов второго порядка (гессиан)
        
        Args:
            sdf: Тензор SDF значений [BS, N]
            points: Тензор точек [BS, N, 3]
            
        Returns:
            Гессиан [BS, N, 3, 3]
        """
        batch_size, num_points = sdf.shape
        
        # Вычисляем первые градиенты
        first_grad = self.compute_sdf_gradient(
            sdf, points, create_graph=True, retain_graph=True
        )
        
        # Вычисляем вторые производные для каждой компоненты
        hessian = []
        
        for dim in range(3):
            grad_component = first_grad[:, :, dim]
            
            second_grad = torch.autograd.grad(
                outputs=grad_component,
                inputs=points,
                grad_outputs=torch.ones_like(grad_component),
                create_graph=False,
                retain_graph=True,
                allow_unused=True
            )[0]
            
            if second_grad is None:
                second_grad = torch.zeros_like(points)
            
            hessian.append(second_grad)
        
        # Комбинируем в гессиан [BS, N, 3, 3]
        hessian = torch.stack(hessian, dim=-1)
        return hessian


class SurfaceProjection:
    """
    Класс для проекции точек на поверхность с использованием SDF
    """
    
    def __init__(self, epsilon: float = 1e-8, max_iterations: int = 5):
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.gradient_calculator = SafeGradientCalculator()
    
    def project_to_surface(
        self,
        model: nn.Module,
        points: torch.Tensor,
        num_iterations: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Итеративная проекция точек на поверхность
        
        Args:
            model: SDF модель
            points: Исходные точки [BS, N, 3]
            num_iterations: Количество итераций проекции
            
        Returns:
            surface_points: Точки на поверхности [BS, N, 3]
            final_sdf: Финальные SDF значения [BS, N]
        """
        if num_iterations is None:
            num_iterations = self.max_iterations
        
        current_points = points.clone()
        
        for iteration in range(num_iterations):
            # Вычисляем SDF и градиенты
            current_points.requires_grad_(True)
            sdf, _ = model.compute_gradients(current_points)
            
            # Вычисляем градиенты безопасно
            gradients = self.gradient_calculator.compute_sdf_gradient(
                sdf, current_points, create_graph=False, retain_graph=False
            )
            
            # Нормализуем градиенты
            grad_norm = F.normalize(gradients, dim=-1, eps=self.epsilon)
            
            # Проекция: p_surface = p - SDF(p) * ∇SDF/‖∇SDF‖
            current_points = current_points - sdf.unsqueeze(-1) * grad_norm
            
            # Detach для экономии памяти (кроме последней итерации)
            if iteration < num_iterations - 1:
                current_points = current_points.detach()
        
        # Финальное вычисление SDF для проектированных точек
        with torch.no_grad():
            final_sdf, _ = model.compute_gradients(current_points)
        
        return current_points, final_sdf
    
    def compute_surface_consistency_loss(
        self,
        model: nn.Module,
        points: torch.Tensor,
        target_on_surface: bool = True
    ) -> torch.Tensor:
        """
        Вычисление loss для консистентности поверхности
        
        Args:
            model: SDF модель
            points: Точки для проверки [BS, N, 3]
            target_on_surface: Целевые точки должны быть на поверхности
            
        Returns:
            Loss значение
        """
        # Проецируем точки на поверхность
        surface_points, surface_sdf = self.project_to_surface(model, points)
        
        if target_on_surface:
            # Для точек на поверхности SDF должен быть близок к 0
            surface_loss = torch.mean(torch.abs(surface_sdf))
        else:
            # Для исходных точек разница должна быть минимальной
            original_sdf, _ = model.compute_gradients(points)
            surface_loss = F.mse_loss(surface_sdf, original_sdf)
        
        return surface_loss
    
    def compute_eikonal_regularization(
        self,
        model: nn.Module,
        points: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисление eikonal regularization
        
        Args:
            model: SDF модель
            points: Точки для regularization [BS, N, 3]
            
        Returns:
            Eikonal loss
        """
        points.requires_grad_(True)
        sdf, gradients = model.compute_gradients(points)
        
        # ‖∇SDF‖ должен быть близок к 1
        grad_norms = torch.norm(gradients, dim=-1)
        eikonal_loss = torch.mean((grad_norms - 1) ** 2)
        
        return eikonal_loss
    
    def compute_curvature_regularization(
        self,
        model: nn.Module,
        points: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисление curvature regularization через вторые производные
        
        Args:
            model: SDF модель
            points: Точки для regularization [BS, N, 3]
            
        Returns:
            Curvature loss
        """
        points.requires_grad_(True)
        sdf, gradients = model.compute_gradients(points, create_graph=True)
        
        # Вычисляем гессиан
        hessian = self.gradient_calculator.compute_second_order_gradients(sdf, points)
        
        # Curvature regularization (минимизация изменения градиентов)
        curvature_loss = torch.mean(torch.norm(hessian, dim=(-1, -2)))
        
        return curvature_loss