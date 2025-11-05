#!/usr/bin/env python3
"""
Скрипт для обучения модели реконструкции поверхностей
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch

# Добавляем путь к src для импорта модулей
sys.path.append(str(Path(__file__).parent.parent))

from src.models.surface_reconstruction import SurfaceReconstructionModel
from src.training.trainer import SurfaceReconstructionTrainer
from src.utils.data_loader import SurfaceDataLoader
from src.utils.visualization import TrainingVisualizer
from src.utils.metrics import ReconstructionMetrics


def setup_logging(log_dir: Path, level: str = "INFO"):
    """Настройка системы логирования"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Train 3D Surface Reconstruction Model')
    
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to processed data directory')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Path to results directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--validate_only', action='store_true',
                       help='Run validation only')
    
    return parser.parse_args()


def setup_environment(config: dict, gpu_id: int):
    """Настройка окружения и устройств"""
    # Установка random seed для воспроизводимости
    torch.manual_seed(config.get('random_seed', 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.get('random_seed', 42))
    
    # Настройка устройства
    use_gpu = config.get('use_gpu', True) and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Available memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    return device


def create_model(config: dict, device: torch.device):
    """Создание модели на основе конфигурации"""
    logger = logging.getLogger(__name__)
    logger.info("Creating model...")
    
    model = SurfaceReconstructionModel(config)
    model.to(device)
    
    # Вывод информации о модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model architecture:\n{model}")
    
    return model


def main():
    """Основная функция обучения"""
    args = parse_args()
    
    # Загрузка конфигурации
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Переопределение параметров из командной строки
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    
    # Настройка путей
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Настройка логирования
    log_level = "DEBUG" if args.debug else config.get('log_level', 'INFO')
    logger = setup_logging(results_dir / 'logs', log_level)
    
    logger.info("Starting 3D Surface Reconstruction Training")
    logger.info(f"Arguments: {args}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Results directory: {results_dir}")
    
    try:
        # Настройка окружения
        device = setup_environment(config, args.gpu)
        
        # Создание модели
        model = create_model(config, device)
        
        # Загрузка данных
        logger.info("Loading data...")
        data_loader = SurfaceDataLoader(
            data_dir=data_dir,
            batch_size=config['training']['batch_size'],
            num_points=config['data']['num_points'],
            num_workers=config['training'].get('num_workers', 4),
            validation_split=config['data'].get('validation_split', 0.2),
            use_kdtree=config['data'].get('use_kdtree', True),
            num_local_points=config['data'].get('num_local_points', 256)
        )
        
        train_loader, val_loader = data_loader.get_data_loaders()
        
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        # Создание визуализатора
        visualizer = TrainingVisualizer(
            save_dir=results_dir / 'visualizations',
            enabled=config['training'].get('visualization', True)
        )
        
        # Создание тренера
        trainer = SurfaceReconstructionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config['training'],
            results_dir=results_dir,
            visualizer=visualizer
        )
        
        # Загрузка чекпоинта если указано
        if args.checkpoint:
            logger.info(f"Loading checkpoint: {args.checkpoint}")
            trainer.load_checkpoint(Path(args.checkpoint))
        
        if args.validate_only:
            # Только валидация
            logger.info("Running validation only...")
            val_results = trainer.validate_epoch(0)
            logger.info(f"Validation results: {val_results}")
        else:
            # Запуск обучения
            logger.info("Starting training...")
            trainer.train()
            
            # Сохранение финальной модели
            final_checkpoint = results_dir / 'checkpoints' / 'final_model.pth'
            trainer.save_checkpoint(final_checkpoint)
            logger.info(f"Final model saved to: {final_checkpoint}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()