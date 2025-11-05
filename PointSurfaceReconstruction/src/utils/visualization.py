import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import open3d as o3d
from matplotlib import cm

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """
    Визуализатор процесса обучения
    """
    
    def __init__(
        self,
        save_dir: Path,
        enabled: bool = True,
        max_plots: int = 10
    ):
        self.save_dir = Path(save_dir)
        self.enabled = enabled
        self.max_plots = max_plots
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Цветовая схема
        self.colors = cm.get_cmap('viridis')
        
        logger.info(f"Training visualizer initialized, save dir: {save_dir}")

    def visualize_batch(
        self,
        batch: Dict,
        predictions: torch.Tensor,
        epoch: int,
        mode: str = 'train'
    ):
        """
        Визуализация батча данных и предсказаний
        
        Args:
            batch: Батч данных
            predictions: Предсказания модели
            epoch: Номер эпохи
            mode: Режим ('train' или 'val')
        """
        if not self.enabled or epoch % 5 != 0:  # Визуализируем каждые 5 эпох
            return
        
        try:
            points = batch['points'].detach().cpu().numpy()
            sdf_gt = batch['sdf_values'].detach().cpu().numpy()
            sdf_pred = predictions.detach().cpu().numpy()
            
            # Визуализируем только первый батч
            points = points[0]
            sdf_gt = sdf_gt[0]
            sdf_pred = sdf_pred[0]
            
            # Создаем фигуру
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{mode.capitalize()} Batch Visualization - Epoch {epoch}')
            
            # 1. GT точки с SDF окраской
            self._plot_points_with_sdf(
                axes[0, 0], points, sdf_gt, 'Ground Truth SDF'
            )
            
            # 2. Predicted точки с SDF окраской
            self._plot_points_with_sdf(
                axes[0, 1], points, sdf_pred, 'Predicted SDF'
            )
            
            # 3. Разница SDF
            sdf_diff = np.abs(sdf_pred - sdf_gt)
            self._plot_points_with_sdf(
                axes[1, 0], points, sdf_diff, 'SDF Absolute Error'
            )
            
            # 4. Surface points (SDF близко к 0)
            surface_mask_gt = np.abs(sdf_gt) < 0.01
            surface_mask_pred = np.abs(sdf_pred) < 0.01
            
            axes[1, 1].scatter(
                points[surface_mask_gt, 0], points[surface_mask_gt, 1],
                c='green', s=1, alpha=0.5, label='GT Surface'
            )
            axes[1, 1].scatter(
                points[surface_mask_pred, 0], points[surface_mask_pred, 1],
                c='red', s=1, alpha=0.5, label='Pred Surface'
            )
            axes[1, 1].set_title('Surface Points Comparison')
            axes[1, 1].legend()
            axes[1, 1].set_aspect('equal')
            
            plt.tight_layout()
            plt.savefig(self.save_dir / f'{mode}_batch_epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Batch visualization failed: {e}")

    def _plot_points_with_sdf(self, ax, points: np.ndarray, sdf: np.ndarray, title: str):
        """Визуализация точек с SDF окраской"""
        scatter = ax.scatter(
            points[:, 0], points[:, 1],
            c=sdf, cmap='viridis', s=2,
            vmin=-0.1, vmax=0.1
        )
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax)

    def plot_training_progress(self, history: Dict, epoch: int):
        """
        Визуализация прогресса обучения
        
        Args:
            history: История обучения
            epoch: Номер эпохи
        """
        if not self.enabled:
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress - Epoch {epoch}')
            
            # Loss curves
            if history['train_loss']:
                epochs_range = range(1, len(history['train_loss']) + 1)
                
                # Total loss
                axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', alpha=0.7)
                axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', alpha=0.7)
                axes[0, 0].set_title('Total Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # SDF metrics
                if history['train_metrics']:
                    train_sdf_mae = [m.get('sdf_mae', 0) for m in history['train_metrics']]
                    val_sdf_mae = [m.get('sdf_mae', 0) for m in history['val_metrics']]
                    
                    axes[0, 1].plot(epochs_range, train_sdf_mae, 'b-', label='Train MAE', alpha=0.7)
                    axes[0, 1].plot(epochs_range, val_sdf_mae, 'r-', label='Val MAE', alpha=0.7)
                    axes[0, 1].set_title('SDF MAE')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('MAE')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
                
                # Surface consistency
                if history['train_metrics']:
                    train_surface = [m.get('surface_consistency', 0) for m in history['train_metrics']]
                    val_surface = [m.get('surface_consistency', 0) for m in history['val_metrics']]
                    
                    axes[1, 0].plot(epochs_range, train_surface, 'b-', label='Train Surface', alpha=0.7)
                    axes[1, 0].plot(epochs_range, val_surface, 'r-', label='Val Surface', alpha=0.7)
                    axes[1, 0].set_title('Surface Consistency')
                    axes[1, 0].set_xlabel('Epoch')
                    axes[1, 0].set_ylabel('Consistency')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                
                # Eikonal loss
                if history['train_metrics']:
                    train_eikonal = [m.get('eikonal_loss', 0) for m in history['train_metrics']]
                    val_eikonal = [m.get('eikonal_loss', 0) for m in history['val_metrics']]
                    
                    axes[1, 1].plot(epochs_range, train_eikonal, 'b-', label='Train Eikonal', alpha=0.7)
                    axes[1, 1].plot(epochs_range, val_eikonal, 'r-', label='Val Eikonal', alpha=0.7)
                    axes[1, 1].set_title('Eikonal Loss')
                    axes[1, 1].set_xlabel('Epoch')
                    axes[1, 1].set_ylabel('Loss')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / f'training_progress_epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Training progress visualization failed: {e}")

    def visualize_3d_reconstruction(
        self,
        points: torch.Tensor,
        sdf_values: torch.Tensor,
        surface_points: Optional[torch.Tensor] = None,
        epoch: int = 0,
        prefix: str = 'reconstruction'
    ):
        """
        3D визуализация реконструкции
        
        Args:
            points: Исходные точки
            sdf_values: SDF значения
            surface_points: Точки на поверхности
            epoch: Номер эпохи
            prefix: Префикс для имени файла
        """
        if not self.enabled:
            return
        
        try:
            points_np = points.detach().cpu().numpy()
            sdf_np = sdf_values.detach().cpu().numpy()
            
            # Создаем point cloud с SDF окраской
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            
            # Нормализуем SDF для цвета
            sdf_normalized = (sdf_np - sdf_np.min()) / (sdf_np.max() - sdf_np.min())
            colors = self.colors(sdf_normalized)[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Сохраняем point cloud
            o3d.io.write_point_cloud(
                str(self.save_dir / f'{prefix}_epoch_{epoch:04d}.ply'),
                pcd
            )
            
            # Визуализируем surface points если есть
            if surface_points is not None:
                surface_np = surface_points.detach().cpu().numpy()
                surface_pcd = o3d.geometry.PointCloud()
                surface_pcd.points = o3d.utility.Vector3dVector(surface_np)
                surface_pcd.paint_uniform_color([1, 0, 0])  # Красный цвет
                
                o3d.io.write_point_cloud(
                    str(self.save_dir / f'{prefix}_surface_epoch_{epoch:04d}.ply'),
                    surface_pcd
                )
                
            logger.debug(f"3D visualization saved for epoch {epoch}")
            
        except Exception as e:
            logger.warning(f"3D visualization failed: {e}")


class PointCloudVisualizer:
    """
    Специализированный визуализатор для точечных облаков
    """
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def compare_point_clouds(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        title: str = "comparison",
        epoch: int = 0
    ):
        """
        Сравнение оригинального и реконструированного точечных облаков
        
        Args:
            original: Оригинальные точки
            reconstructed: Реконструированные точки
            title: Заголовок
            epoch: Номер эпохи
        """
        try:
            original_np = original.detach().cpu().numpy()
            reconstructed_np = reconstructed.detach().cpu().numpy()
            
            # Создаем point clouds
            pcd_orig = o3d.geometry.PointCloud()
            pcd_orig.points = o3d.utility.Vector3dVector(original_np)
            pcd_orig.paint_uniform_color([0, 0, 1])  # Синий
            
            pcd_recon = o3d.geometry.PointCloud()
            pcd_recon.points = o3d.utility.Vector3dVector(reconstructed_np)
            pcd_recon.paint_uniform_color([1, 0, 0])  # Красный
            
            # Сохраняем оба облака
            o3d.io.write_point_cloud(
                str(self.save_dir / f'{title}_original_epoch_{epoch:04d}.ply'),
                pcd_orig
            )
            o3d.io.write_point_cloud(
                str(self.save_dir / f'{title}_reconstructed_epoch_{epoch:04d}.ply'),
                pcd_recon
            )
            
        except Exception as e:
            logger.warning(f"Point cloud comparison failed: {e}")

    def create_combined_visualization(
        self,
        points_list: List[torch.Tensor],
        colors_list: List[List[float]],
        names: List[str],
        filename: str
    ):
        """
        Создание комбинированной визуализации нескольких точечных облаков
        
        Args:
            points_list: Список точечных облаков
            colors_list: Список цветов для каждого облака
            names: Имена облаков
            filename: Имя файла для сохранения
        """
        try:
            combined_pcd = o3d.geometry.PointCloud()
            
            all_points = []
            all_colors = []
            
            for points, color, name in zip(points_list, colors_list, names):
                points_np = points.detach().cpu().numpy()
                all_points.append(points_np)
                
                colors_array = np.tile(np.array(color).reshape(1, 3), (len(points_np), 1))
                all_colors.append(colors_array)
                
                logger.debug(f"Added {len(points_np)} points for {name}")
            
            # Объединяем все точки
            combined_points = np.vstack(all_points)
            combined_colors = np.vstack(all_colors)
            
            combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
            combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
            
            o3d.io.write_point_cloud(str(self.save_dir / filename), combined_pcd)
            
        except Exception as e:
            logger.warning(f"Combined visualization failed: {e}")