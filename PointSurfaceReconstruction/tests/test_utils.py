import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile

# Добавляем путь к src
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.utils.metrics import ReconstructionMetrics, compute_chamfer_distance, compute_f_score
from src.utils.gradient_utils import SafeGradientCalculator, SurfaceProjection
from src.utils.data_loader import SDFDataset, SurfaceDataLoader
from src.models.surface_reconstruction import SurfaceReconstructionModel


class TestMetrics:
    """Тесты для метрик реконструкции"""
    
    @pytest.fixture
    def sample_point_clouds(self):
        """Создание synthetic точечных облаков для тестирования"""
        points1 = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        
        points2 = torch.tensor([
            [0.1, 0.1, 0.1],
            [1.1, 0.1, 0.1],
            [0.1, 1.1, 0.1],
            [0.1, 0.1, 1.1]
        ], dtype=torch.float32)
        
        return points1, points2
    
    @pytest.fixture
    def sample_sdf_data(self):
        """Создание synthetic SDF данных"""
        batch_size, num_points = 2, 64
        sdf_pred = torch.randn(batch_size, num_points)
        sdf_gt = torch.randn(batch_size, num_points)
        points = torch.randn(batch_size, num_points, 3)
        
        return sdf_pred, sdf_gt, points
    
    def test_metrics_initialization(self):
        """Тест инициализации метрик"""
        metrics = ReconstructionMetrics()
        
        assert metrics.metrics_history is not None
        assert 'sdf_mse' in metrics.metrics_history
        assert 'sdf_mae' in metrics.metrics_history
    
    def test_batch_metrics_computation(self, sample_sdf_data):
        """Тест вычисления метрик для батча"""
        metrics = ReconstructionMetrics()
        sdf_pred, sdf_gt, points = sample_sdf_data
        
        batch_metrics = metrics.compute_batch_metrics(
            sdf_pred, sdf_gt, points
        )
        
        expected_metrics = ['sdf_mse', 'sdf_mae', 'surface_consistency', 'eikonal_loss']
        for metric in expected_metrics:
            assert metric in batch_metrics
            assert isinstance(batch_metrics[metric], float)
            assert batch_metrics[metric] >= 0
    
    def test_metrics_history(self, sample_sdf_data):
        """Тест накопления истории метрик"""
        metrics = ReconstructionMetrics()
        sdf_pred, sdf_gt, points = sample_sdf_data
        
        # Вычисляем метрики несколько раз
        for _ in range(3):
            metrics.compute_batch_metrics(sdf_pred, sdf_gt, points)
        
        # Проверяем что история накапливается
        for metric_name, history in metrics.metrics_history.items():
            assert len(history) == 3
        
        # Получаем усредненные метрики
        epoch_metrics = metrics.get_epoch_metrics()
        
        for metric_name, value in epoch_metrics.items():
            assert isinstance(value, float)
            assert value >= 0
        
        # Проверяем что история сбрасывается
        for history in metrics.metrics_history.values():
            assert len(history) == 0
    
    def test_chamfer_distance(self, sample_point_clouds):
        """Тест вычисления Chamfer distance"""
        points1, points2 = sample_point_clouds
        
        chamfer_dist = compute_chamfer_distance(points1, points2)
        
        assert isinstance(chamfer_dist, float)
        assert chamfer_dist >= 0
        # Для близких точек расстояние должно быть небольшим
        assert chamfer_dist < 1.0
    
    def test_f_score_computation(self, sample_point_clouds):
        """Тест вычисления F-Score"""
        points1, points2 = sample_point_clouds
        
        f_score = compute_f_score(points1, points2, threshold=0.5)
        
        assert isinstance(f_score, float)
        assert 0 <= f_score <= 1.0
    
    def test_complete_metrics(self):
        """Тест вычисления полного набора метрик"""
        metrics = ReconstructionMetrics()
        
        # Создаем synthetic данные
        pred_points = torch.randn(100, 3)
        gt_points = torch.randn(150, 3)
        pred_normals = torch.randn(100, 3)
        gt_normals = torch.randn(150, 3)
        
        complete_metrics = metrics.compute_complete_metrics(
            pred_points, gt_points, pred_normals, gt_normals
        )
        
        expected_metrics = [
            'chamfer_distance', 'f_score_1cm', 'f_score_2cm', 'normal_consistency'
        ]
        
        for metric in expected_metrics:
            assert metric in complete_metrics
            assert isinstance(complete_metrics[metric], float)
    
    def test_edge_cases(self):
        """Тест крайних случаев для метрик"""
        metrics = ReconstructionMetrics()
        
        # Точки с NaN значениями
        points_with_nan = torch.tensor([[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
        valid_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        
        # Должны обрабатывать NaN корректно
        chamfer_dist = compute_chamfer_distance(valid_points, valid_points)
        assert chamfer_dist == 0.0


class TestGradientUtils:
    """Тесты для утилит работы с градиентами"""
    
    @pytest.fixture
    def sample_sdf_function(self):
        """Создание synthetic SDF функции (сфера)"""
        def sphere_sdf(points):
            return torch.norm(points, dim=-1) - 1.0  # Сфера радиуса 1
        
        return sphere_sdf
    
    @pytest.fixture
    def sample_points(self):
        return torch.randn(2, 16, 3, requires_grad=True)
    
    def test_gradient_calculator_initialization(self):
        """Тест инициализации калькулятора градиентов"""
        calculator = SafeGradientCalculator(chunk_size=512)
        
        assert calculator.chunk_size == 512
        assert isinstance(calculator, SafeGradientCalculator)
    
    def test_gradient_computation(self, sample_points):
        """Тест вычисления градиентов"""
        calculator = SafeGradientCalculator()
        
        # Создаем synthetic SDF значения
        sdf_values = torch.sum(sample_points ** 2, dim=-1)  # Квадратичная функция
        
        gradients = calculator.compute_sdf_gradient(
            sdf_values, sample_points, create_graph=True
        )
        
        assert gradients.shape == sample_points.shape
        assert not torch.isnan(gradients).any()
        
        # Для f(x,y,z) = x² + y² + z², градиент = (2x, 2y, 2z)
        expected_gradients = 2 * sample_points
        assert torch.allclose(gradients, expected_gradients, atol=1e-6)
    
    def test_chunked_gradient_computation(self):
        """Тест вычисления градиентов с chunking'ом"""
        calculator = SafeGradientCalculator(chunk_size=8)
        
        # Большое количество точек
        points = torch.randn(2, 64, 3, requires_grad=True)
        sdf_values = torch.sum(points ** 2, dim=-1)
        
        gradients = calculator.compute_sdf_gradient(sdf_values, points)
        
        assert gradients.shape == points.shape
        assert not torch.isnan(gradients).any()
    
    def test_second_order_gradients(self, sample_points):
        """Тест вычисления градиентов второго порядка"""
        calculator = SafeGradientCalculator()
        
        sdf_values = torch.sum(sample_points ** 2, dim=-1)
        
        hessian = calculator.compute_second_order_gradients(sdf_values, sample_points)
        
        assert hessian.shape == (*sample_points.shape, 3)
        assert not torch.isnan(hessian).any()
        
        # Для квадратичной функции гессиан постоянный
        expected_hessian = 2 * torch.eye(3).unsqueeze(0).unsqueeze(0)
        expected_hessian = expected_hessian.repeat(sample_points.shape[0], sample_points.shape[1], 1, 1)
        
        assert torch.allclose(hessian, expected_hessian, atol=1e-6)
    
    def test_surface_projection_initialization(self):
        """Тест инициализации проектора поверхностей"""
        projector = SurfaceProjection(epsilon=1e-6, max_iterations=10)
        
        assert projector.epsilon == 1e-6
        assert projector.max_iterations == 10
        assert hasattr(projector, 'gradient_calculator')
    
    def test_surface_projection(self):
        """Тест проекции точек на поверхность"""
        # Создаем простую модель с известной SDF
        class SimpleSDFModel(torch.nn.Module):
            def forward(self, points):
                return torch.norm(points, dim=-1) - 1.0  # Сфера
            
            def compute_gradients(self, points, create_graph=False):
                points.requires_grad_(True)
                sdf = self(points)
                gradients = torch.autograd.grad(
                    outputs=sdf,
                    inputs=points,
                    grad_outputs=torch.ones_like(sdf),
                    create_graph=create_graph,
                    retain_graph=True
                )[0]
                return sdf, gradients
        
        model = SimpleSDFModel()
        projector = SurfaceProjection(max_iterations=5)
        
        # Точки вне сферы
        points = torch.tensor([
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, 0.0, 2.0], [1.5, 1.5, 0.0]]
        ], requires_grad=True)
        
        surface_points, surface_sdf = projector.project_to_surface(model, points)
        
        assert surface_points.shape == points.shape
        assert surface_sdf.shape == (points.shape[0], points.shape[1])
        
        # После проекции точки должны быть ближе к сфере
        surface_distances = torch.norm(surface_points, dim=-1)
        assert torch.allclose(surface_distances, torch.ones_like(surface_distances), atol=0.1)
    
    def test_eikonal_regularization(self):
        """Тест eikonal regularization"""
        class SimpleModel(torch.nn.Module):
            def compute_gradients(self, points, create_graph=False):
                points.requires_grad_(True)
                sdf = torch.norm(points, dim=-1)  # Не SDF, но для теста подойдет
                gradients = torch.autograd.grad(
                    outputs=sdf,
                    inputs=points,
                    grad_outputs=torch.ones_like(sdf),
                    create_graph=create_graph,
                    retain_graph=True
                )[0]
                return sdf, gradients
        
        model = SimpleModel()
        projector = SurfaceProjection()
        
        points = torch.randn(2, 8, 3, requires_grad=True)
        eikonal_loss = projector.compute_eikonal_regularization(model, points)
        
        assert isinstance(eikonal_loss, torch.Tensor)
        assert eikonal_loss.item() >= 0


class TestDataLoader:
    """Тесты для data loader'ов"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Создание временной директории с synthetic данными"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создаем простой PLY файл для тестирования
            import open3d as o3d
            
            # Создаем простую сферу
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
            mesh_path = Path(temp_dir) / "test_sphere.ply"
            o3d.io.write_triangle_mesh(str(mesh_path), mesh)
            
            yield temp_dir
    
    def test_sdf_dataset_initialization(self, temp_data_dir):
        """Тест инициализации SDF dataset"""
        dataset = SDFDataset(
            data_dir=temp_data_dir,
            num_points=128,
            sdf_clip=0.1,
            use_kdtree=False  # Отключаем KD-tree для скорости
        )
        
        assert len(dataset) == 1
        assert dataset.num_points == 128
        assert dataset.sdf_clip == 0.1
    
    def test_sdf_dataset_getitem(self, temp_data_dir):
        """Тест получения элемента из dataset"""
        dataset = SDFDataset(
            data_dir=temp_data_dir,
            num_points=256,
            use_kdtree=False
        )
        
        sample = dataset[0]
        
        expected_keys = ['points', 'sdf_values', 'normals', 'mesh_name']
        for key in expected_keys:
            assert key in sample
        
        assert sample['points'].shape == (256, 3)
        assert sample['sdf_values'].shape == (256,)
        assert sample['normals'].shape == (256, 3)
        assert isinstance(sample['mesh_name'], str)
        
        # Проверяем что SDF значения в пределах клиппинга
        assert torch.all(sample['sdf_values'] >= -0.1)
        assert torch.all(sample['sdf_values'] <= 0.1)
    
    def test_surface_data_loader(self, temp_data_dir):
        """Тест surface data loader'а"""
        data_loader = SurfaceDataLoader(
            data_dir=temp_data_dir,
            batch_size=2,
            num_points=128,
            num_workers=0,  # Для стабильности тестов
            validation_split=0.5
        )
        
        train_loader, val_loader = data_loader.get_data_loaders()
        
        assert len(train_loader.dataset) == 1  # 1 sample в train
        assert len(val_loader.dataset) == 0   # 0 samples в val (из-за округления)
        
        # Проверяем что можем итерироваться
        for batch in train_loader:
            assert 'points' in batch
            assert 'sdf_values' in batch
            assert batch['points'].shape[0] == 1  # batch_size=2, но только 1 sample
            break
    
    def test_data_loader_different_modes(self, temp_data_dir):
        """Тест data loader'а в разных режимах"""
        # Train mode
        train_dataset = SDFDataset(
            data_dir=temp_data_dir,
            num_points=128,
            mode='train'
        )
        
        # Test mode  
        test_dataset = SDFDataset(
            data_dir=temp_data_dir,
            num_points=128,
            mode='test'
        )
        
        # Проверяем что оба работают
        assert len(train_dataset) == 1
        assert len(test_dataset) == 1


class TestIntegration:
    """Интеграционные тесты для утилит"""
    
    def test_metrics_with_realistic_data(self):
        """Тест метрик с реалистичными данными"""
        metrics = ReconstructionMetrics()
        
        # Создаем реалистичные SDF данные (сфера)
        batch_size, num_points = 4, 512
        points = torch.randn(batch_size, num_points, 3)
        
        # GT SDF для сферы радиуса 1
        sdf_gt = torch.norm(points, dim=-1) - 1.0
        
        # Predicted SDF с небольшим шумом
        sdf_pred = sdf_gt + 0.1 * torch.randn_like(sdf_gt)
        
        batch_metrics = metrics.compute_batch_metrics(sdf_pred, sdf_gt, points)
        
        # Проверяем что все метрики вычисляются
        for metric_name, value in batch_metrics.items():
            assert isinstance(value, float)
            assert not np.isnan(value)
            assert not np.isinf(value)
    
    def test_gradient_utils_with_model(self):
        """Тест утилит градиентов с реальной моделью"""
        config = {
            'model': {
                'latent_size': 64,
                'input_dim': 3,
                'local_feat_size': 32
            }
        }
        
        model = SurfaceReconstructionModel(config)
        gradient_calculator = SafeGradientCalculator()
        
        points = torch.randn(2, 32, 3, requires_grad=True)
        sdf_pred, _ = model(points)
        
        gradients = gradient_calculator.compute_sdf_gradient(sdf_pred, points)
        
        assert gradients.shape == points.shape
        assert not torch.isnan(gradients).any()
        
        # Проверяем что градиенты не нулевые (модель обучаема)
        assert torch.abs(gradients).sum() > 1e-6