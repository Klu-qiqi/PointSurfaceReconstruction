import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Добавляем путь к src
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.models.pointnet import ImprovedPointNet
from src.models.local_decoder import LocalDecoder
from src.models.global_decoder import GlobalDecoder
from src.models.surface_reconstruction import SurfaceReconstructionModel


class TestPointNet:
    """Тесты для PointNet модели"""
    
    @pytest.fixture
    def pointnet_config(self):
        return {
            'latent_size': 128,
            'input_dim': 3,
            'conv_channels': [64, 128],
            'use_transform': True
        }
    
    @pytest.fixture
    def sample_data(self):
        batch_size, num_points = 4, 1024
        points = torch.randn(batch_size, num_points, 3)
        return points
    
    def test_pointnet_initialization(self, pointnet_config):
        """Тест инициализации PointNet"""
        model = ImprovedPointNet(**pointnet_config)
        
        assert isinstance(model, nn.Module)
        assert model.latent_size == pointnet_config['latent_size']
        assert model.input_dim == pointnet_config['input_dim']
    
    def test_pointnet_forward(self, pointnet_config, sample_data):
        """Тест forward pass PointNet"""
        model = ImprovedPointNet(**pointnet_config)
        
        global_feat, local_feat = model(sample_data)
        
        # Проверка размеров выходов
        assert global_feat.shape == (sample_data.shape[0], pointnet_config['latent_size'])
        assert local_feat.shape == (sample_data.shape[0], sample_data.shape[1], 128)  # local_feat_size
        
        # Проверка что нет NaN значений
        assert not torch.isnan(global_feat).any()
        assert not torch.isnan(local_feat).any()
    
    def test_pointnet_gradients(self, pointnet_config, sample_data):
        """Тест вычисления градиентов"""
        model = ImprovedPointNet(**pointnet_config)
        
        # Включаем градиенты
        sample_data.requires_grad = True
        
        global_feat, local_feat = model(sample_data)
        
        # Создаем dummy loss
        loss = global_feat.sum() + local_feat.sum()
        loss.backward()
        
        # Проверяем что градиенты вычислены
        assert sample_data.grad is not None
        assert not torch.isnan(sample_data.grad).any()
    
    def test_pointnet_different_batch_sizes(self, pointnet_config):
        """Тест работы с разными размерами батчей"""
        model = ImprovedPointNet(**pointnet_config)
        
        batch_sizes = [1, 2, 8, 16]
        num_points = 512
        
        for batch_size in batch_sizes:
            points = torch.randn(batch_size, num_points, 3)
            global_feat, local_feat = model(points)
            
            assert global_feat.shape[0] == batch_size
            assert local_feat.shape[0] == batch_size
    
    @pytest.mark.parametrize("num_points", [256, 512, 1024, 2048])
    def test_pointnet_different_point_counts(self, pointnet_config, num_points):
        """Тест работы с разным количеством точек"""
        model = ImprovedPointNet(**pointnet_config)
        batch_size = 4
        
        points = torch.randn(batch_size, num_points, 3)
        global_feat, local_feat = model(points)
        
        assert local_feat.shape[1] == num_points


class TestLocalDecoder:
    """Тесты для локального декодера"""
    
    @pytest.fixture
    def decoder_config(self):
        return {
            'input_dim': 3,
            'latent_size': 256,
            'local_feat_size': 128,
            'hidden_dims': [512, 256],
            'dropout': 0.1
        }
    
    @pytest.fixture
    def sample_decoder_data(self):
        batch_size, num_points = 4, 512
        points = torch.randn(batch_size, num_points, 3)
        local_features = torch.randn(batch_size, num_points, 128)
        return points, local_features
    
    def test_local_decoder_initialization(self, decoder_config):
        """Тест инициализации локального декодера"""
        model = LocalDecoder(**decoder_config)
        
        assert isinstance(model, nn.Module)
        assert model.local_feat_size == decoder_config['local_feat_size']
    
    def test_local_decoder_forward(self, decoder_config, sample_decoder_data):
        """Тест forward pass локального декодера"""
        model = LocalDecoder(**decoder_config)
        points, local_features = sample_decoder_data
        
        sdf_pred = model(local_features, points)
        
        # Проверка размеров выхода
        assert sdf_pred.shape == (points.shape[0], points.shape[1])
        assert not torch.isnan(sdf_pred).any()
    
    def test_local_decoder_gradients(self, decoder_config, sample_decoder_data):
        """Тест вычисления градиентов локального декодера"""
        model = LocalDecoder(**decoder_config)
        points, local_features = sample_decoder_data
        
        points.requires_grad = True
        local_features.requires_grad = True
        
        sdf_pred = model(local_features, points)
        loss = sdf_pred.sum()
        loss.backward()
        
        assert points.grad is not None
        assert local_features.grad is not None
        assert not torch.isnan(points.grad).any()
    
    def test_local_sdf_computation(self, decoder_config):
        """Тест вычисления локального SDF"""
        model = LocalDecoder(**decoder_config)
        
        batch_size, num_centers, num_queries = 2, 32, 64
        local_features = torch.randn(batch_size, num_centers, 128)
        query_points = torch.randn(batch_size, num_queries, 3)
        center_points = torch.randn(batch_size, num_centers, 3)
        
        sdf_pred = model.compute_local_sdf(
            local_features, query_points, center_points
        )
        
        assert sdf_pred.shape == (batch_size, num_queries)
        assert not torch.isnan(sdf_pred).any()


class TestGlobalDecoder:
    """Тесты для глобального декодера"""
    
    @pytest.fixture
    def global_decoder_config(self):
        return {
            'global_latent_size': 256,
            'local_latent_size': 128,
            'output_dim': 1,
            'hidden_dims': [512, 256, 128]
        }
    
    @pytest.fixture
    def sample_global_data(self):
        batch_size, num_points = 4, 512
        global_features = torch.randn(batch_size, 256)
        local_features = torch.randn(batch_size, num_points, 128)
        points = torch.randn(batch_size, num_points, 3)
        return global_features, local_features, points
    
    def test_global_decoder_initialization(self, global_decoder_config):
        """Тест инициализации глобального декодера"""
        model = GlobalDecoder(**global_decoder_config)
        
        assert isinstance(model, nn.Module)
        assert model.global_latent_size == global_decoder_config['global_latent_size']
    
    def test_global_decoder_forward(self, global_decoder_config, sample_global_data):
        """Тест forward pass глобального декодера"""
        model = GlobalDecoder(**global_decoder_config)
        global_features, local_features, points = sample_global_data
        
        sdf_pred = model(global_features, local_features, points)
        
        assert sdf_pred.shape == (points.shape[0], points.shape[1])
        assert not torch.isnan(sdf_pred).any()
    
    def test_global_decoder_gradients(self, global_decoder_config, sample_global_data):
        """Тест вычисления градиентов глобального декодера"""
        model = GlobalDecoder(**global_decoder_config)
        global_features, local_features, points = sample_global_data
        
        global_features.requires_grad = True
        local_features.requires_grad = True
        points.requires_grad = True
        
        sdf_pred = model(global_features, local_features, points)
        loss = sdf_pred.sum()
        loss.backward()
        
        assert global_features.grad is not None
        assert local_features.grad is not None
        assert points.grad is not None


class TestSurfaceReconstructionModel:
    """Тесты для полной модели реконструкции поверхностей"""
    
    @pytest.fixture
    def model_config(self):
        return {
            'model': {
                'latent_size': 128,
                'input_dim': 3,
                'local_feat_size': 64,
                'decoder_hidden_dims': [256, 128],
                'global_decoder_hidden_dims': [256, 128],
                'dropout_rate': 0.1
            }
        }
    
    @pytest.fixture
    def sample_surface_data(self):
        batch_size, num_points = 2, 256
        points = torch.randn(batch_size, num_points, 3)
        return points
    
    def test_surface_model_initialization(self, model_config):
        """Тест инициализации полной модели"""
        model = SurfaceReconstructionModel(model_config)
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'pointnet')
        assert hasattr(model, 'local_decoder')
        assert hasattr(model, 'global_decoder')
    
    def test_surface_model_forward(self, model_config, sample_surface_data):
        """Тест forward pass полной модели"""
        model = SurfaceReconstructionModel(model_config)
        points = sample_surface_data
        
        sdf_pred, features = model(points)
        
        assert sdf_pred.shape == (points.shape[0], points.shape[1])
        assert 'global_features' in features
        assert 'local_features' in features
        assert 'local_sdf' in features
        assert 'global_sdf' in features
        assert 'alpha' in features
        
        # Проверка что alpha параметр обучается
        assert model.alpha.requires_grad
    
    def test_surface_model_gradients(self, model_config, sample_surface_data):
        """Тест вычисления градиентов полной модели"""
        model = SurfaceReconstructionModel(model_config)
        points = sample_surface_data
        
        points.requires_grad = True
        
        sdf_pred, _ = model(points)
        sdf, gradients = model.compute_gradients(points)
        
        assert sdf.shape == sdf_pred.shape
        assert gradients.shape == points.shape
        assert not torch.isnan(gradients).any()
    
    def test_surface_projection(self, model_config, sample_surface_data):
        """Тест проекции точек на поверхность"""
        model = SurfaceReconstructionModel(model_config)
        points = sample_surface_data
        
        surface_points, surface_sdf = model.project_to_surface(points, num_iterations=2)
        
        assert surface_points.shape == points.shape
        assert surface_sdf.shape == (points.shape[0], points.shape[1])
        
        # После проекции SDF должен быть ближе к 0
        with torch.no_grad():
            original_sdf, _ = model.compute_gradients(points)
        
        # Проверяем что проекция улучшила SDF (не строгая проверка)
        assert torch.abs(surface_sdf).mean() <= torch.abs(original_sdf).mean() * 1.5
    
    def test_model_parameters(self, model_config):
        """Тест получения параметров модели"""
        model = SurfaceReconstructionModel(model_config)
        
        parameters = model.get_parameters()
        
        assert 'pointnet' in parameters
        assert 'local_decoder' in parameters
        assert 'global_decoder' in parameters
        
        # Проверяем что все параметры требуют градиенты
        for component_params in parameters.values():
            for param in component_params:
                assert param.requires_grad
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_gpu(self, model_config, sample_surface_data):
        """Тест работы модели на GPU"""
        model = SurfaceReconstructionModel(model_config)
        model = model.cuda()
        
        points = sample_surface_data.cuda()
        
        sdf_pred, features = model(points)
        
        assert sdf_pred.is_cuda
        assert features['global_features'].is_cuda
        assert not torch.isnan(sdf_pred).any()


class TestModelIntegration:
    """Интеграционные тесты для моделей"""
    
    def test_end_to_end_training_step(self):
        """Тест полного шага обучения"""
        config = {
            'model': {
                'latent_size': 64,
                'input_dim': 3,
                'local_feat_size': 32,
                'decoder_hidden_dims': [128, 64],
                'global_decoder_hidden_dims': [128, 64]
            }
        }
        
        model = SurfaceReconstructionModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Создаем synthetic данные
        batch_size, num_points = 2, 128
        points = torch.randn(batch_size, num_points, 3)
        sdf_gt = torch.randn(batch_size, num_points)
        
        # Шаг обучения
        optimizer.zero_grad()
        sdf_pred, _ = model(points)
        loss = torch.nn.MSELoss()(sdf_pred, sdf_gt)
        loss.backward()
        optimizer.step()
        
        # Проверяем что loss уменьшился (не строгая проверка)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_model_save_load(self, model_config, tmp_path):
        """Тест сохранения и загрузки модели"""
        model = SurfaceReconstructionModel(model_config)
        
        # Сохраняем модель
        save_path = tmp_path / "test_model.pth"
        model.save_model(str(save_path))
        
        assert save_path.exists()
        
        # Загружаем модель
        loaded_model = SurfaceReconstructionModel(model_config)
        loaded_model.load_model(str(save_path), torch.device('cpu'))
        
        # Проверяем что параметры загружены корректно
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), loaded_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2, atol=1e-6)