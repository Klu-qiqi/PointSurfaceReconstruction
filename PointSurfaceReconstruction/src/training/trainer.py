"""
Тренер для обучения модели реконструкции поверхностей
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from ..utils.metrics import ReconstructionMetrics
from ..utils.visualization import TrainingVisualizer


class SurfaceReconstructionTrainer:
    """
    Тренер для обучения pipeline реконструкции поверхностей
    """
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Dict[str, Any],
        results_dir: Path,
        visualizer: Optional[TrainingVisualizer] = None
    ):
        self.models = models
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.results_dir = results_dir
        self.visualizer = visualizer
        
        self.logger = logging.getLogger(__name__)
        self.metrics = ReconstructionMetrics()
        
        # Настройка оптимизаторов
        self.setup_optimizers()
        
        # История обучения
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
    def setup_optimizers(self):
        """Настройка оптимизаторов для всех компонентов модели"""
        parameters = []
        for model in self.models.values():
            parameters.extend(model.parameters())
        
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-6)
        )
        
        # Scheduler с warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config['learning_rate'],
            epochs=self.config['num_epochs'],
            steps_per_epoch=len(self.train_loader)
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Одна эпоха обучения"""
        self.set_models_mode(train=True)
        
        epoch_loss = 0.0
        epoch_metrics = {}
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Перенос данных на устройство
            points = batch['points'].to(self.device)
            sdf_gt = batch['sdf_values'].to(self.device)
            normals = batch.get('normals', None)
            if normals is not None:
                normals = normals.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # PointNet features
            global_feat, local_feat = self.models['pointnet'](points)
            
            # Local decoder
            local_sdf = self.models['local_decoder'](local_feat, points)
            
            # Global decoder
            global_sdf = self.models['global_decoder'](global_feat, local_feat, points)
            
            # Combined prediction
            sdf_pred = local_sdf + global_sdf
            
            # Compute losses
            losses = self.compute_losses(
                sdf_pred=sdf_pred,
                sdf_gt=sdf_gt,
                points=points,
                normals=normals
            )
            
            total_loss = losses['total']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.get_all_parameters(),
                self.config.get('grad_clip', 1.0)
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            epoch_loss += total_loss.item()
            batch_metrics = self.metrics.compute_batch_metrics(sdf_pred, sdf_gt)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss.item(),
                'lr': self.scheduler.get_last_lr()[0]
            })
            
            # Визуализация первых нескольких батчей
            if batch_idx == 0 and self.visualizer is not None:
                self.visualizer.visualize_batch(
                    batch, sdf_pred, epoch, 'train'
                )
        
        # Compute epoch averages
        avg_loss = epoch_loss / len(self.train_loader)
        avg_metrics = self.metrics.get_epoch_metrics()
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_metrics'].append(avg_metrics)
        
        return {
            'loss': avg_loss,
            'metrics': avg_metrics
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Валидация после эпохи"""
        self.set_models_mode(train=False)
        
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                points = batch['points'].to(self.device)
                sdf_gt = batch['sdf_values'].to(self.device)
                normals = batch.get('normals', None)
                if normals is not None:
                    normals = normals.to(self.device)
                
                # Forward pass
                global_feat, local_feat = self.models['pointnet'](points)
                local_sdf = self.models['local_decoder'](local_feat, points)
                global_sdf = self.models['global_decoder'](global_feat, local_feat, points)
                sdf_pred = local_sdf + global_sdf
                
                losses = self.compute_losses(
                    sdf_pred=sdf_pred,
                    sdf_gt=sdf_gt,
                    points=points,
                    normals=normals
                )
                
                val_loss += losses['total'].item()
                self.metrics.compute_batch_metrics(sdf_pred, sdf_gt)
                
                # Визуализация первого батча
                if batch_idx == 0 and self.visualizer is not None:
                    self.visualizer.visualize_batch(
                        batch, sdf_pred, epoch, 'val'
                    )
        
        avg_val_loss = val_loss / len(self.val_loader)
        val_metrics = self.metrics.get_epoch_metrics()
        
        self.history['val_loss'].append(avg_val_loss)
        self.history['val_metrics'].append(val_metrics)
        
        return {
            'loss': avg_val_loss,
            'metrics': val_metrics
        }
    
    def compute_losses(
        self,
        sdf_pred: torch.Tensor,
        sdf_gt: torch.Tensor,
        points: torch.Tensor,
        normals: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Вычисление всех лоссов"""
        losses = {}
        
        # Основной SDF loss
        losses['sdf_loss'] = nn.MSELoss()(sdf_pred, sdf_gt)
        
        # Eikonal regularization
        losses['eikonal_loss'] = self.compute_eikonal_loss(sdf_pred, points)
        
        # Surface consistency loss
        losses['surface_loss'] = self.compute_surface_consistency_loss(sdf_pred, points)
        
        # Normal consistency loss если есть GT нормали
        if normals is not None:
            losses['normal_loss'] = self.compute_normal_consistency_loss(sdf_pred, points, normals)
        
        # Total loss с весами
        total_loss = 0.0
        loss_weights = self.config.get('loss_weights', {})
        
        for loss_name, loss_value in losses.items():
            weight = loss_weights.get(loss_name, 1.0)
            total_loss += weight * loss_value
        
        losses['total'] = total_loss
        
        return losses
    
    def compute_eikonal_loss(self, sdf: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Eikonal regularization для SDF"""
        points.requires_grad_(True)
        
        grad = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # ‖∇SDF‖ должен быть близок к 1
        eikonal_loss = torch.mean((torch.norm(grad, dim=-1) - 1) ** 2)
        return eikonal_loss
    
    def compute_surface_consistency_loss(self, sdf: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Loss для консистентности поверхности"""
        # Для точек около поверхности SDF должен быть малым
        surface_mask = torch.abs(sdf) < 0.01
        if torch.any(surface_mask):
            return torch.mean(torch.abs(sdf[surface_mask]))
        return torch.tensor(0.0, device=sdf.device)
    
    def compute_normal_consistency_loss(
        self,
        sdf: torch.Tensor,
        points: torch.Tensor,
        normals_gt: torch.Tensor
    ) -> torch.Tensor:
        """Loss для консистентности нормалей"""
        points.requires_grad_(True)
        
        grad = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Нормализованные градиенты SDF должны совпадать с GT нормалями
        grad_norm = F.normalize(grad, dim=-1)
        normal_loss = 1 - torch.abs(torch.sum(grad_norm * normals_gt, dim=-1))
        
        return torch.mean(normal_loss)
    
    def train(self):
        """Основной цикл обучения"""
        self.logger.info("Starting training loop...")
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            self.logger.info(f"Epoch {epoch}/{self.config['num_epochs']}")
            
            # Обучение
            train_results = self.train_epoch(epoch)
            
            # Валидация
            val_results = self.validate_epoch(epoch)
            
            # Логирование
            self.log_epoch_results(epoch, train_results, val_results)
            
            # Сохранение чекпоинта
            if epoch % self.config.get('checkpoint_frequency', 10) == 0:
                self.save_checkpoint(self.results_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pth')
            
            # Early stopping
            if self.check_early_stopping(val_results['loss']):
                self.logger.info("Early stopping triggered")
                break
            
            # Визуализация прогресса
            if self.visualizer is not None:
                self.visualizer.plot_training_progress(self.history, epoch)
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """Проверка условия early stopping"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
            # Сохраняем лучшую модель
            self.save_checkpoint(self.results_dir / 'checkpoints' / 'best_model.pth')
        else:
            self.epochs_no_improve += 1
        
        patience = self.config.get('patience', 50)
        return self.epochs_no_improve >= patience
    
    def set_models_mode(self, train: bool = True):
        """Установка режима моделей (train/eval)"""
        for model in self.models.values():
            model.train() if train else model.eval()
    
    def get_all_parameters(self):
        """Получение всех параметров моделей"""
        parameters = []
        for model in self.models.values():
            parameters.extend(model.parameters())
        return parameters
    
    def save_checkpoint(self, checkpoint_path: Path):
        """Сохранение чекпоинта"""
        checkpoint = {
            'epoch': len(self.history['train_loss']),
            'models_state_dict': {name: model.state_dict() for name, model in self.models.items()},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Загрузка чекпоинта"""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Загрузка состояний моделей
        for name, model in self.models.items():
            model.load_state_dict(checkpoint['models_state_dict'][name])
        
        # Загрузка состояний оптимизаторов
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Загрузка истории
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {checkpoint['epoch']}")
    
    def log_epoch_results(self, epoch: int, train_results: dict, val_results: dict):
        """Логирование результатов эпохи"""
        self.logger.info(
            f"Epoch {epoch}: "
            f"Train Loss: {train_results['loss']:.6f}, "
            f"Val Loss: {val_results['loss']:.6f}, "
            f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
        )