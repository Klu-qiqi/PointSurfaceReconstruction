import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class ReconstructionMetrics:
    """
    Класс для вычисления метрик качества реконструкции
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Сброс накопленных метрик"""
        self.metrics_history = {
            'sdf_mse': [],
            'sdf_mae': [],
            'surface_consistency': [],
            'normal_consistency': [],
            'eikonal_loss': []
        }
    
    def compute_batch_metrics(
        self,
        sdf_pred: torch.Tensor,
        sdf_gt: torch.Tensor,
        points: Optional[torch.Tensor] = None,
        normals_pred: Optional[torch.Tensor] = None,
        normals_gt: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Вычисление метрик для батча
        
        Args:
            sdf_pred: Предсказанные SDF значения [BS, N]
            sdf_gt: GT SDF значения [BS, N]
            points: Точки [BS, N, 3]
            normals_pred: Предсказанные нормали [BS, N, 3]
            normals_gt: GT нормали [BS, N, 3]
            
        Returns:
            Словарь с метриками
        """
        metrics = {}
        
        # SDF метрики
        metrics['sdf_mse'] = F.mse_loss(sdf_pred, sdf_gt).item()
        metrics['sdf_mae'] = F.l1_loss(sdf_pred, sdf_gt).item()
        
        # Surface consistency (SDF близко к 0 для точек на поверхности)
        surface_mask = torch.abs(sdf_gt) < 0.01
        if torch.any(surface_mask):
            metrics['surface_consistency'] = torch.mean(torch.abs(sdf_pred[surface_mask])).item()
        else:
            metrics['surface_consistency'] = 0.0
        
        # Normal consistency если есть GT нормали
        if normals_pred is not None and normals_gt is not None:
            cos_similarity = F.cosine_similarity(normals_pred, normals_gt, dim=-1)
            metrics['normal_consistency'] = torch.mean(cos_similarity).item()
        
        # Eikonal regularization если есть точки
        if points is not None:
            eikonal_loss = self._compute_eikonal_loss(sdf_pred, points)
            metrics['eikonal_loss'] = eikonal_loss.item()
        
        # Сохраняем метрики
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        return metrics
    
    def _compute_eikonal_loss(self, sdf: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Вычисление eikonal loss"""
        points.requires_grad_(True)
        
        grad = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True
        )[0]
        
        eikonal_loss = torch.mean((torch.norm(grad, dim=-1) - 1) ** 2)
        return eikonal_loss
    
    def get_epoch_metrics(self) -> Dict[str, float]:
        """Получение усредненных метрик за эпоху"""
        epoch_metrics = {}
        
        for metric_name, values in self.metrics_history.items():
            if values:
                epoch_metrics[metric_name] = np.mean(values)
        
        self.reset()
        return epoch_metrics
    
    def compute_chamfer_distance(
        self,
        points1: torch.Tensor,
        points2: torch.Tensor
    ) -> float:
        """
        Вычисление Chamfer Distance между двумя наборами точек
        
        Args:
            points1: Точки [N, 3]
            points2: Точки [M, 3]
            
        Returns:
            Chamfer distance
        """
        if isinstance(points1, torch.Tensor):
            points1 = points1.detach().cpu().numpy()
        if isinstance(points2, torch.Tensor):
            points2 = points2.detach().cpu().numpy()
        
        # Build KD-trees для эффективного поиска
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        
        # Расстояния от points1 до points2
        dist1, _ = tree2.query(points1)
        # Расстояния от points2 до points1
        dist2, _ = tree1.query(points2)
        
        chamfer_dist = np.mean(dist1) + np.mean(dist2)
        return float(chamfer_dist)
    
    def compute_f_score(
        self,
        points1: torch.Tensor,
        points2: torch.Tensor,
        threshold: float = 0.01
    ) -> float:
        """
        Вычисление F-Score между двумя наборами точек
        
        Args:
            points1: Точки [N, 3]
            points2: Точки [M, 3]
            threshold: Порог для определения совпадения
            
        Returns:
            F-Score
        """
        if isinstance(points1, torch.Tensor):
            points1 = points1.detach().cpu().numpy()
        if isinstance(points2, torch.Tensor):
            points2 = points2.detach().cpu().numpy()
        
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        
        # Precision: какая доля points1 близка к points2
        dist1, _ = tree2.query(points1)
        precision = np.mean(dist1 < threshold)
        
        # Recall: какая доля points2 близка к points1
        dist2, _ = tree1.query(points2)
        recall = np.mean(dist2 < threshold)
        
        if precision + recall == 0:
            return 0.0
        
        f_score = 2 * precision * recall / (precision + recall)
        return float(f_score)
    
    def compute_complete_metrics(
        self,
        predicted_points: torch.Tensor,
        ground_truth_points: torch.Tensor,
        predicted_normals: Optional[torch.Tensor] = None,
        ground_truth_normals: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Вычисление полного набора метрик реконструкции
        
        Args:
            predicted_points: Предсказанные точки [N, 3]
            ground_truth_points: GT точки [M, 3]
            predicted_normals: Предсказанные нормали [N, 3]
            ground_truth_normals: GT нормали [M, 3]
            
        Returns:
            Словарь с метриками
        """
        metrics = {}
        
        # Chamfer Distance
        metrics['chamfer_distance'] = self.compute_chamfer_distance(
            predicted_points, ground_truth_points
        )
        
        # F-Score с разными порогами
        metrics['f_score_1cm'] = self.compute_f_score(
            predicted_points, ground_truth_points, threshold=0.01
        )
        metrics['f_score_2cm'] = self.compute_f_score(
            predicted_points, ground__truth_points, threshold=0.02
        )
        
        # Normal consistency если есть нормали
        if predicted_normals is not None and ground_truth_normals is not None:
            # Для вычисления consistency нужно сопоставить точки
            if isinstance(predicted_points, torch.Tensor):
                predicted_points_np = predicted_points.detach().cpu().numpy()
                ground_truth_points_np = ground_truth_points.detach().cpu().numpy()
            else:
                predicted_points_np = predicted_points
                ground_truth_points_np = ground_truth_points
            
            tree = cKDTree(ground_truth_points_np)
            _, indices = tree.query(predicted_points_np)
            
            # Берем соответствующие нормали
            if isinstance(ground_truth_normals, torch.Tensor):
                gt_normals_matched = ground_truth_normals[indices].detach().cpu().numpy()
                pred_normals_np = predicted_normals.detach().cpu().numpy()
            else:
                gt_normals_matched = ground_truth_normals[indices]
                pred_normals_np = predicted_normals
            
            # Cosine similarity
            cos_sim = np.sum(pred_normals_np * gt_normals_matched, axis=1)
            metrics['normal_consistency'] = float(np.mean(cos_sim))
        
        return metrics


def compute_chamfer_distance(
    points1: torch.Tensor,
    points2: torch.Tensor
) -> float:
    """Упрощенная функция для вычисления Chamfer Distance"""
    metrics = ReconstructionMetrics()
    return metrics.compute_chamfer_distance(points1, points2)


def compute_f_score(
    points1: torch.Tensor,
    points2: torch.Tensor,
    threshold: float = 0.01
) -> float:
    """Упрощенная функция для вычисления F-Score"""
    metrics = ReconstructionMetrics()
    return metrics.compute_f_score(points1, points2, threshold)