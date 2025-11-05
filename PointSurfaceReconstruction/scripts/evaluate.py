#!/usr/bin/env python3
"""
Скрипт для оценки обученной модели
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from src.models.surface_reconstruction import SurfaceReconstructionModel
from src.utils.data_loader import SurfaceDataLoader
from src.utils.metrics import ReconstructionMetrics
from src.utils.visualization import PointCloudVisualizer


def setup_evaluation(config_path: str, checkpoint_path: str, results_dir: Path):
    """Настройка оценки"""
    # Загрузка конфигурации
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Настройка устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создание модели
    model = SurfaceReconstructionModel(config)
    model.load_model(checkpoint_path, device)
    model.eval()
    
    # Создание визуализатора
    visualizer = PointCloudVisualizer(results_dir / 'evaluation')
    
    return model, device, visualizer, config


def evaluate_model(model, data_loader, device, visualizer, results_dir):
    """Оценка модели на тестовых данных"""
    logger = logging.getLogger(__name__)
    metrics_calculator = ReconstructionMetrics()
    
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            points = batch['points'].to(device)
            sdf_gt = batch['sdf_values'].to(device)
            normals_gt = batch.get('normals', None)
            if normals_gt is not None:
                normals_gt = normals_gt.to(device)
            
            # Forward pass
            sdf_pred, features = model(points)
            
            # Вычисление метрик
            batch_metrics = metrics_calculator.compute_batch_metrics(
                sdf_pred, sdf_gt, points, None, normals_gt
            )
            all_metrics.append(batch_metrics)
            
            # Визуализация первых нескольких батчей
            if batch_idx < 3:
                # Проецируем точки на поверхность
                surface_points, surface_sdf = model.project_to_surface(points)
                
                # Визуализация
                visualizer.compare_point_clouds(
                    points[0], surface_points[0],
                    title=f"reconstruction_batch_{batch_idx}",
                    epoch=0
                )
            
            logger.info(f"Processed batch {batch_idx + 1}/{len(data_loader)}")
    
    # Усреднение метрик
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_metrics


def main():
    """Основная функция оценки"""
    parser = argparse.ArgumentParser(description='Evaluate trained surface reconstruction model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test data')
    parser.add_argument('--results_dir', type=str, default='evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Настройка
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(results_dir / 'evaluation.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        model, device, visualizer, config = setup_evaluation(
            args.config, args.checkpoint, results_dir
        )
        
        # Загрузка тестовых данных
        test_loader = SurfaceDataLoader(
            data_dir=args.data_dir,
            batch_size=config['training']['batch_size'],
            num_points=config['data']['num_points'],
            mode='test'
        ).get_test_loader()
        
        logger.info("Starting evaluation...")
        metrics = evaluate_model(model, test_loader, device, visualizer, results_dir)
        
        # Сохранение результатов
        logger.info("Evaluation results:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.6f}")
        
        with open(results_dir / 'metrics.json', 'w') as f:
            import json
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Results saved to {results_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()