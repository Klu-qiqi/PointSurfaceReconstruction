import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import tempfile

# Добавляем путь к src
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.training.trainer import SurfaceReconstructionTrainer
from src.training.loss_functions import SurfaceReconstructionLoss, AdaptiveLossWeightScheduler
from src.models.surface_reconstruction import SurfaceReconstructionModel
from src.utils.data_loader import SurfaceDataLoader


class TestLossFunctions:
    """Тесты для функций потерь"""
    
    @pytest.fixture
    def sample_training_data(self):
        batch_size, num_points = 2, 128
        sdf_pred = torch.randn(batch_size, num_points)
        sdf_gt = torch.randn(batch_size, num_points)
        points = torch.randn(batch_size, num_points, 3)
        normals_pred = torch.randn(batch_size, num_points, 3)
        normals_gt = torch.randn(batch_size, num_points, 3)
        
        return sdf_pred, sdf_gt, points, normals_pred, normals_gt
    
    @pytest.fixture
    def loss_weights(self):
        return {
            'sdf_loss': 1.0,
            'surface_loss': 0.1,
            'normal_loss': 0.01,
            'eikonal_loss': 0.1,
            'curvature_loss': 0.001
        }
    
    def test_loss_function_initialization(self, loss_weights):
        """Тест инициализации функции потерь"""
        loss_fn = SurfaceReconstructionLoss(loss_weights)
        
        assert loss_fn.loss_weights == loss_weights
        assert isinstance(loss_fn.mse_loss, nn.MSELoss)
        assert isinstance(loss_fn.l1_loss, nn.L1Loss)
    
    def test_loss_computation_basic(self, loss_weights, sample_training_data):
        """Тест базового вычисления потерь"""
        loss_fn = SurfaceReconstructionLoss(loss_weights)
        sdf_pred, sdf_gt, points, _, _ = sample_training_data
        
        total_loss, loss_dict = loss_fn(sdf_pred, sdf_gt, points)
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() >= 0
        
        expected_losses = ['sdf_loss', 'surface_loss', 'eikonal_loss', 'curvature_loss', 'total_loss']
        for loss_name in expected_losses:
            assert loss_name in loss_dict
            assert isinstance(loss_dict[loss_name], torch.Tensor)
    
    def test_loss_computation_with_normals(self, loss_weights, sample_training_data):
        """Тест вычисления потерь с нормалями"""
        loss_fn = SurfaceReconstructionLoss(loss_weights)
        sdf_pred, sdf_gt, points, normals_pred, normals_gt = sample_training_data
        
        total_loss, loss_dict = loss_fn(
            sdf_pred, sdf_gt, points, normals_pred, normals_gt
        )
        
        assert 'normal_loss' in loss_dict
        assert loss_dict['normal_loss'].item() >= 0
    
    def test_loss_computation_with_gradients(self, loss_weights, sample_training_data):
        """Тест вычисления потерь с предварительно вычисленными градиентами"""
        loss_fn = SurfaceReconstructionLoss(loss_weights)
        sdf_pred, sdf_gt, points, _, _ = sample_training_data
        
        # Вычисляем градиенты
        points.requires_grad_(True)
        gradients = torch.autograd.grad(
            outputs=sdf_pred,
            inputs=points,
            grad_outputs=torch.ones_like(sdf_pred),
            create_graph=True
        )[0]
        
        total_loss, loss_dict = loss_fn(
            sdf_pred, sdf_gt, points, gradients_pred=gradients
        )
        
        assert 'eikonal_loss' in loss_dict
        assert loss_dict['eikonal_loss'].item() >= 0
    
    def test_surface_consistency_loss(self, loss_weights):
        """Тест surface consistency loss"""
        loss_fn = SurfaceReconstructionLoss(loss_weights)
        
        # Точки на поверхности (SDF близко к 0)
        sdf_pred = torch.tensor([[0.01, -0.005, 0.02, 0.5, -0.3]])
        sdf_gt = torch.tensor([[0.0, 0.0, 0.0, 0.5, -0.3]])
        
        surface_loss = loss_fn._compute_surface_consistency_loss(sdf_pred, sdf_gt)
        
        assert isinstance(surface_loss, torch.Tensor)
        assert surface_loss.item() >= 0
    
    def test_normal_consistency_loss(self, loss_weights):
        """Тест normal consistency loss"""
        loss_fn = SurfaceReconstructionLoss(loss_weights)
        
        normals_pred = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        normals_gt = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        
        normal_loss = loss_fn._compute_normal_consistency_loss(normals_pred, normals_gt)
        
        # Для одинаковых нормалей loss должен быть близок к 0
        assert normal_loss.item() < 0.1
    
    def test_eikonal_loss(self, loss_weights):
        """Тест eikonal loss"""
        loss_fn = SurfaceReconstructionLoss(loss_weights)
        
        # Градиенты с нормой 1 (идеальный случай)
        gradients = torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ])
        
        eikonal_loss = loss_fn._compute_eikonal_loss(gradients)
        
        assert eikonal_loss.item() < 1e-6  # Должен быть очень маленьким
    
    def test_adaptive_loss_scheduler(self):
        """Тест адаптивного scheduler'а весов"""
        initial_weights = {
            'sdf_loss': 1.0,
            'surface_loss': 0.1,
            'eikonal_loss': 0.1
        }
        
        scheduler = AdaptiveLossWeightScheduler(initial_weights, update_frequency=10)
        
        # Первое обновление
        current_losses = {'sdf_loss': 0.5, 'surface_loss': 0.2, 'eikonal_loss': 0.8}
        weights1 = scheduler.update_weights(current_losses, iteration=5)
        
        # Веса не должны измениться до update_frequency
        assert weights1 == initial_weights
        
        # Обновление на iteration=10
        weights2 = scheduler.update_weights(current_losses, iteration=10)
        
        # Веса должны измениться
        assert weights2 != initial_weights
        for weight in weights2.values():
            assert weight >= 0.1  # Минимальный вес


class TestTrainer:
    """Тесты для тренера"""
    
    @pytest.fixture
    def sample_trainer_components(self):
        """Создание компонентов для тестирования тренера"""
        config = {
            'model': {
                'latent_size': 64,
                'input_dim': 3,
                'local_feat_size': 32
            }
        }
        
        model = SurfaceReconstructionModel(config)
        
        # Создаем synthetic data loader
        class MockDataLoader:
            def __init__(self):
                self.dataset = [None] * 4  # 4 samples
            
            def __len__(self):
                return 2  # 2 batches
            
            def __iter__(self):
                for _ in range(2):
                    yield {
                        'points': torch.randn(2, 64, 3),
                        'sdf_values': torch.randn(2, 64),
                        'normals': torch.randn(2, 64, 3)
                    }
        
        train_loader = MockDataLoader()
        val_loader = MockDataLoader()
        
        device = torch.device('cpu')
        
        training_config = {
            'batch_size': 2,
            'learning_rate': 1e-4,
            'num_epochs': 3,
            'weight_decay': 1e-6,
            'grad_clip': 1.0,
            'loss_weights': {
                'sdf_loss': 1.0,
                'surface_loss': 0.1,
                'eikonal_loss': 0.1
            }
        }
        
        return model, train_loader, val_loader, device, training_config
    
    def test_trainer_initialization(self, sample_trainer_components):
        """Тест инициализации тренера"""
        model, train_loader, val_loader, device, training_config = sample_trainer_components
        
        trainer = SurfaceReconstructionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=training_config,
            results_dir=Path("test_results")
        )
        
        assert trainer.model == model
        assert trainer.train_loader == train_loader
        assert trainer.val_loader == val_loader
        assert trainer.device == device
        assert trainer.config == training_config
        
        # Проверяем что оптимизатор создан
        assert hasattr(trainer, 'optimizer')
        assert isinstance(trainer.optimizer, torch.optim.Optimizer)
        
        # Проверяем что scheduler создан
        assert hasattr(trainer, 'scheduler')
    
    def test_loss_computation(self, sample_trainer_components):
        """Тест вычисления потерь в тренере"""
        model, train_loader, val_loader, device, training_config = sample_trainer_components
        trainer = SurfaceReconstructionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=training_config,
            results_dir=Path("test_results")
        )
        
        # Создаем synthetic данные
        sdf_pred = torch.randn(2, 64)
        sdf_gt = torch.randn(2, 64)
        points = torch.randn(2, 64, 3)
        normals = torch.randn(2, 64, 3)
        
        losses = trainer.compute_losses(sdf_pred, sdf_gt, points, normals)
        
        expected_losses = ['sdf_loss', 'surface_loss', 'normal_loss', 'eikonal_loss', 'total']
        for loss_name in expected_losses:
            assert loss_name in losses
            assert isinstance(losses[loss_name], torch.Tensor)
            assert losses[loss_name].item() >= 0
    
    def test_eikonal_loss_computation(self, sample_trainer_components):
        """Тест вычисления eikonal loss"""
        model, train_loader, val_loader, device, training_config = sample_trainer_components
        trainer = SurfaceReconstructionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=training_config,
            results_dir=Path("test_results")
        )
        
        sdf = torch.randn(2, 64)
        points = torch.randn(2, 64, 3, requires_grad=True)
        
        eikonal_loss = trainer.compute_eikonal_loss(sdf, points)
        
        assert isinstance(eikonal_loss, torch.Tensor)
        assert eikonal_loss.item() >= 0
    
    def test_surface_consistency_loss_computation(self, sample_trainer_components):
        """Тест вычисления surface consistency loss"""
        model, train_loader, val_loader, device, training_config = sample_trainer_components
        trainer = SurfaceReconstructionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=training_config,
            results_dir=Path("test_results")
        )
        
        sdf = torch.tensor([[0.01, -0.005, 0.02, 0.5, -0.3]])
        points = torch.randn(1, 5, 3)
        
        surface_loss = trainer.compute_surface_consistency_loss(sdf, points)
        
        assert isinstance(surface_loss, torch.Tensor)
        assert surface_loss.item() >= 0
    
    def test_normal_consistency_loss_computation(self, sample_trainer_components):
        """Тест вычисления normal consistency loss"""
        model, train_loader, val_loader, device, training_config = sample_trainer_components
        trainer = SurfaceReconstructionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=training_config,
            results_dir=Path("test_results")
        )
        
        sdf = torch.randn(1, 8)
        points = torch.randn(1, 8, 3, requires_grad=True)
        normals_gt = torch.randn(1, 8, 3)
        
        # Вычисляем градиенты (как предсказанные нормали)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True
        )[0]
        
        normal_loss = trainer.compute_normal_consistency_loss(sdf, points, normals_gt)
        
        assert isinstance(normal_loss, torch.Tensor)
        assert normal_loss.item() >= 0
    
    def test_train_epoch(self, sample_trainer_components):
        """Тест одной эпохи обучения"""
        model, train_loader, val_loader, device, training_config = sample_trainer_components
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = SurfaceReconstructionTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                config=training_config,
                results_dir=Path(temp_dir)
            )
            
            # Запускаем одну эпоху обучения
            train_results = trainer.train_epoch(1)
            
            assert 'loss' in train_results
            assert 'metrics' in train_results
            assert isinstance(train_results['loss'], float)
            assert isinstance(train_results['metrics'], dict)
            
            # Проверяем что история обновляется
            assert len(trainer.history['train_loss']) == 1
            assert len(trainer.history['train_metrics']) == 1
    
    def test_validation_epoch(self, sample_trainer_components):
        """Тест одной эпохи валидации"""
        model, train_loader, val_loader, device, training_config = sample_trainer_components
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = SurfaceReconstructionTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                config=training_config,
                results_dir=Path(temp_dir)
            )
            
            # Запускаем валидацию
            val_results = trainer.validate_epoch(1)
            
            assert 'loss' in val_results
            assert 'metrics' in val_results
            assert isinstance(val_results['loss'], float)
            assert isinstance(val_results['metrics'], dict)
            
            # Проверяем что история обновляется
            assert len(trainer.history['val_loss']) == 1
            assert len(trainer.history['val_metrics']) == 1
    
    def test_early_stopping(self, sample_trainer_components):
        """Тест early stopping"""
        model, train_loader, val_loader, device, training_config = sample_trainer_components
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = SurfaceReconstructionTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                config=training_config,
                results_dir=Path(temp_dir)
            )
            
            # Симулируем улучшение валидационного loss
            should_stop = trainer.check_early_stopping(0.5)  # Хороший loss
            assert not should_stop
            assert trainer.epochs_no_improve == 0
            
            # Симулируем ухудшение
            should_stop = trainer.check_early_stopping(0.6)  # Хуже чем лучший (0.5)
            assert not should_stop
            assert trainer.epochs_no_improve == 1
            
            # Много ухудшений подряд
            trainer.epochs_no_improve = training_config['patience'] - 1
            should_stop = trainer.check_early_stopping(0.7)
            assert should_stop
    
    def test_checkpoint_save_load(self, sample_trainer_components):
        """Тест сохранения и загрузки чекпоинтов"""
        model, train_loader, val_loader, device, training_config = sample_trainer_components
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            trainer1 = SurfaceReconstructionTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                config=training_config,
                results_dir=temp_path
            )
            
            # Сохраняем чекпоинт
            checkpoint_path = temp_path / "test_checkpoint.pth"
            trainer1.save_checkpoint(checkpoint_path)
            
            assert checkpoint_path.exists()
            
            # Создаем нового тренера и загружаем чекпоинт
            trainer2 = SurfaceReconstructionTrainer(
                model=SurfaceReconstructionModel(model.config),
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                config=training_config,
                results_dir=temp_path
            )
            
            trainer2.load_checkpoint(checkpoint_path)
            
            # Проверяем что история загружена
            assert trainer2.history['train_loss'] == trainer1.history['train_loss']
            assert trainer2.best_val_loss == trainer1.best_val_loss


class TestTrainingIntegration:
    """Интеграционные тесты обучения"""
    
    def test_complete_training_cycle(self):
        """Тест полного цикла обучения (сокращенный)"""
        config = {
            'model': {
                'latent_size': 32,
                'input_dim': 3,
                'local_feat_size': 16,
                'decoder_hidden_dims': [64, 32],
                'global_decoder_hidden_dims': [64, 32]
            }
        }
        
        model = SurfaceReconstructionModel(config)
        
        # Создаем synthetic data loader
        class SimpleDataLoader:
            def __init__(self, num_batches=2):
                self.num_batches = num_batches
                self.dataset = [None] * 4
            
            def __len__(self):
                return self.num_batches
            
            def __iter__(self):
                for _ in range(self.num_batches):
                    yield {
                        'points': torch.randn(2, 32, 3),
                        'sdf_values': torch.randn(2, 32),
                        'normals': torch.randn(2, 32, 3)
                    }
        
        train_loader = SimpleDataLoader()
        val_loader = SimpleDataLoader(1)
        
        training_config = {
            'batch_size': 2,
            'learning_rate': 1e-4,
            'num_epochs': 2,  # Только 2 эпохи для теста
            'weight_decay': 1e-6,
            'grad_clip': 1.0,
            'patience': 10,
            'checkpoint_frequency': 1,
            'loss_weights': {
                'sdf_loss': 1.0,
                'surface_loss': 0.1,
                'eikonal_loss': 0.1
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = SurfaceReconstructionTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=torch.device('cpu'),
                config=training_config,
                results_dir=Path(temp_dir)
            )
            
            # Запускаем сокращенное обучение
            trainer.train()
            
            # Проверяем что обучение прошло
            assert len(trainer.history['train_loss']) == 2
            assert len(trainer.history['val_loss']) == 2
            
            # Проверяем что чекпоинты созданы
            checkpoint_dir = Path(temp_dir) / 'checkpoints'
            assert checkpoint_dir.exists()
            
            # Должен быть создан лучший чекпоинт
            best_checkpoint = checkpoint_dir / 'best_model.pth'
            assert best_checkpoint.exists()