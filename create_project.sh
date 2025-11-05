#!/bin/bash

# Создание основной структуры папок
mkdir -p PointSurfaceReconstruction/configs
mkdir -p PointSurfaceReconstruction/src/models
mkdir -p PointSurfaceReconstruction/src/training
mkdir -p PointSurfaceReconstruction/src/utils
mkdir -p PointSurfaceReconstruction/src/data
mkdir -p PointSurfaceReconstruction/scripts
mkdir -p PointSurfaceReconstruction/tests
mkdir -p PointSurfaceReconstruction/data/raw
mkdir -p PointSurfaceReconstruction/data/processed
mkdir -p PointSurfaceReconstruction/results/checkpoints
mkdir -p PointSurfaceReconstruction/results/logs
mkdir -p PointSurfaceReconstruction/results/visualizations

# Создание файлов конфигурации
cat > PointSurfaceReconstruction/configs/training_config.yaml << 'EOF'
# Конфигурация обучения
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  validation_split: 0.2
  
data:
  point_cloud_size: 1024
  surface_samples: 5000
  
optimizer:
  name: "adam"
  weight_decay: 0.0001

scheduler:
  name: "cosine"
  warmup_epochs: 5
EOF

cat > PointSurfaceReconstruction/configs/model_config.yaml << 'EOF'
# Конфигурация модели
pointnet:
  input_dim: 3
  feature_dim: 1024
  layers: [64, 128, 1024]

global_decoder:
  input_dim: 1024
  hidden_dims: [512, 256, 128]
  output_dim: 3

local_decoder:
  input_dim: 1024 + 3  # features + query point
  hidden_dims: [256, 128, 64]
  output_dim: 1  # SDF value
EOF

# Создание __init__.py файлов
touch PointSurfaceReconstruction/src/__init__.py
touch PointSurfaceReconstruction/src/models/__init__.py
touch PointSurfaceReconstruction/src/training/__init__.py
touch PointSurfaceReconstruction/src/utils/__init__.py
touch PointSurfaceReconstruction/src/data/__init__.py
touch PointSurfaceReconstruction/tests/__init__.py

# Создание файлов моделей
cat > PointSurfaceReconstruction/src/models/pointnet.py << 'EOF'
import torch
import torch.nn as nn

class PointNet(nn.Module):
    def __init__(self, input_dim=3, feature_dim=1024):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, feature_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_dim)
        
    def forward(self, x):
        # x: (batch_size, num_points, 3)
        x = x.transpose(2, 1)  # (batch_size, 3, num_points)
        
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, x.shape[1])
        
        return x
EOF

cat > PointSurfaceReconstruction/src/models/global_decoder.py << 'EOF'
import torch.nn as nn

class GlobalDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(GlobalDecoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
EOF

cat > PointSurfaceReconstruction/src/models/local_decoder.py << 'EOF'
import torch.nn as nn

class LocalDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(LocalDecoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, features, query_points):
        # features: (batch_size, feature_dim)
        # query_points: (batch_size, num_queries, 3)
        batch_size, num_queries, _ = query_points.shape
        
        # Repeat features for each query point
        features = features.unsqueeze(1).repeat(1, num_queries, 1)
        
        # Concatenate features with query points
        x = torch.cat([features, query_points], dim=-1)
        x = x.view(batch_size * num_queries, -1)
        
        # Pass through network
        output = self.network(x)
        output = output.view(batch_size, num_queries, -1)
        
        return output
EOF

# Создание файлов обучения
cat > PointSurfaceReconstruction/src/training/trainer.py << 'EOF'
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['optimizer']['weight_decay']
        )
        
        self.criterion = torch.nn.MSELoss()
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (point_cloud, sdf_values) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_sdf = self.model(point_cloud)
            loss = self.criterion(pred_sdf, sdf_values)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for point_cloud, sdf_values in self.val_loader:
                pred_sdf = self.model(point_cloud)
                loss = self.criterion(pred_sdf, sdf_values)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.6f}')
            print(f'  Val Loss: {val_loss:.6f}')
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        os.makedirs('results/checkpoints', exist_ok=True)
        torch.save(checkpoint, f'results/checkpoints/checkpoint_epoch_{epoch}.pth')
EOF

cat > PointSurfaceReconstruction/src/training/loss_functions.py << 'EOF'
import torch
import torch.nn as nn

class SDFLoss(nn.Module):
    def __init__(self):
        super(SDFLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred_sdf, gt_sdf, weights=None):
        if weights is not None:
            return torch.mean(weights * (pred_sdf - gt_sdf) ** 2)
        else:
            return self.mse_loss(pred_sdf, gt_sdf)

class EikonalLoss(nn.Module):
    def __init__(self):
        super(EikonalLoss, self).__init__()
        
    def forward(self, gradients):
        # gradients should have norm 1 for SDF
        grad_norm = torch.norm(gradients, p=2, dim=-1)
        return torch.mean((grad_norm - 1) ** 2)
EOF

# Создание утилит
cat > PointSurfaceReconstruction/src/utils/data_loader.py << 'EOF'
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PointCloudDataset(Dataset):
    def __init__(self, point_clouds, sdf_values):
        self.point_clouds = point_clouds
        self.sdf_values = sdf_values
        
    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        point_cloud = torch.FloatTensor(self.point_clouds[idx])
        sdf_value = torch.FloatTensor(self.sdf_values[idx])
        return point_cloud, sdf_value

def create_data_loader(point_clouds, sdf_values, batch_size=32, shuffle=True):
    dataset = PointCloudDataset(point_clouds, sdf_values)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
EOF

cat > PointSurfaceReconstruction/src/utils/metrics.py << 'EOF'
import torch
import numpy as np

def chamfer_distance(pred_points, gt_points):
    """
    Compute Chamfer distance between two point clouds
    """
    pred_points = pred_points.unsqueeze(1)  # (B, 1, N, 3)
    gt_points = gt_points.unsqueeze(2)      # (B, M, 1, 3)
    
    distances = torch.sum((pred_points - gt_points) ** 2, dim=-1)
    
    min_pred_to_gt = torch.min(distances, dim=1)[0].mean()
    min_gt_to_pred = torch.min(distances, dim=2)[0].mean()
    
    return min_pred_to_gt + min_gt_to_pred

def sdf_accuracy(pred_sdf, gt_sdf, threshold=0.01):
    """
    Compute accuracy of SDF predictions
    """
    errors = torch.abs(pred_sdf - gt_sdf)
    accuracy = torch.mean((errors < threshold).float())
    return accuracy
EOF

cat > PointSurfaceReconstruction/src/utils/visualization.py << 'EOF'
import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_point_cloud(points, title="Point Cloud"):
    """
    Plot 3D point cloud
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return fig

def plot_sdf_slice(sdf_grid, level=0, title="SDF Slice"):
    """
    Plot 2D slice of SDF
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(sdf_grid, cmap='RdBu_r', origin='lower')
    plt.colorbar(label='SDF Value')
    plt.contour(sdf_grid, levels=[level], colors='black')
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()
EOF

# Создание препроцессора данных
cat > PointSurfaceReconstruction/src/data/preprocessor.py << 'EOF'
import numpy as np
import trimesh

class DataPreprocessor:
    def __init__(self, num_points=1024, num_samples=5000):
        self.num_points = num_points
        self.num_samples = num_samples
        
    def load_mesh(self, mesh_path):
        """Load mesh from file"""
        return trimesh.load_mesh(mesh_path)
    
    def sample_point_cloud(self, mesh):
        """Sample point cloud from mesh surface"""
        points, _ = trimesh.sample.sample_surface(mesh, self.num_points)
        return points
    
    def sample_sdf_points(self, mesh, bbox_size=2.0):
        """Sample points in space and compute SDF values"""
        # Sample points in bounding box
        points = np.random.uniform(-bbox_size/2, bbox_size/2, 
                                 (self.num_samples, 3))
        
        # Compute SDF values
        sdf_values = mesh.nearest.signed_distance(points)
        
        return points, sdf_values
    
    def normalize_points(self, points):
        """Normalize points to unit sphere"""
        centroid = np.mean(points, axis=0)
        points = points - centroid
        scale = np.max(np.linalg.norm(points, axis=1))
        points = points / scale
        return points
EOF

# Создание скриптов
cat > PointSurfaceReconstruction/scripts/train.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
from src.models.pointnet import PointNet
from src.training.trainer import Trainer
from src.utils.data_loader import create_data_loader

def main():
    # Load configs
    with open('configs/training_config.yaml', 'r') as f:
        train_config = yaml.safe_load(f)
    
    with open('configs/model_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Initialize model
    model = PointNet(
        input_dim=model_config['pointnet']['input_dim'],
        feature_dim=model_config['pointnet']['feature_dim']
    )
    
    # Create dummy data (replace with actual data loading)
    num_samples = 100
    point_clouds = torch.randn(num_samples, 1024, 3)
    sdf_values = torch.randn(num_samples, 5000, 1)
    
    # Create data loaders
    train_loader = create_data_loader(
        point_clouds[:80], sdf_values[:80],
        batch_size=train_config['training']['batch_size']
    )
    val_loader = create_data_loader(
        point_clouds[80:], sdf_values[80:],
        batch_size=train_config['training']['batch_size'],
        shuffle=False
    )
    
    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, train_config)
    
    # Start training
    trainer.train(epochs=train_config['training']['epochs'])

if __name__ == "__main__":
    main()
EOF

cat > PointSurfaceReconstruction/scripts/evaluate.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml
from src.models.pointnet import PointNet
from src.utils.metrics import chamfer_distance, sdf_accuracy

def main():
    # Load model
    with open('configs/model_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)
    
    model = PointNet(
        input_dim=model_config['pointnet']['input_dim'],
        feature_dim=model_config['pointnet']['feature_dim']
    )
    
    # Load checkpoint
    checkpoint = torch.load('results/checkpoints/checkpoint_epoch_99.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate on test data
    # Add your evaluation code here
    print("Evaluation completed!")

if __name__ == "__main__":
    main()
EOF

cat > PointSurfaceReconstruction/scripts/preprocess_data.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessor import DataPreprocessor
import numpy as np
import os

def main():
    preprocessor = DataPreprocessor(num_points=1024, num_samples=5000)
    
    # Process all mesh files in data/raw
    raw_data_dir = 'data/raw'
    processed_data_dir = 'data/processed'
    
    os.makedirs(processed_data_dir, exist_ok=True)
    
    for mesh_file in os.listdir(raw_data_dir):
        if mesh_file.endswith(('.obj', '.ply', '.stl')):
            mesh_path = os.path.join(raw_data_dir, mesh_file)
            print(f"Processing {mesh_path}")
            
            # Load and process mesh
            mesh = preprocessor.load_mesh(mesh_path)
            point_cloud = preprocessor.sample_point_cloud(mesh)
            sdf_points, sdf_values = preprocessor.sample_sdf_points(mesh)
            
            # Normalize
            point_cloud = preprocessor.normalize_points(point_cloud)
            sdf_points = preprocessor.normalize_points(sdf_points)
            
            # Save processed data
            base_name = os.path.splitext(mesh_file)[0]
            np.save(os.path.join(processed_data_dir, f'{base_name}_points.npy'), point_cloud)
            np.save(os.path.join(processed_data_dir, f'{base_name}_sdf_points.npy'), sdf_points)
            np.save(os.path.join(processed_data_dir, f'{base_name}_sdf_values.npy'), sdf_values)
    
    print("Data preprocessing completed!")

if __name__ == "__main__":
    main()
EOF

# Создание тестов
cat > PointSurfaceReconstruction/tests/test_models.py << 'EOF'
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.pointnet import PointNet
from src.models.local_decoder import LocalDecoder

def test_pointnet():
    model = PointNet(input_dim=3, feature_dim=1024)
    x = torch.randn(4, 1024, 3)  # batch_size=4, num_points=1024, dim=3
    output = model(x)
    assert output.shape == (4, 1024)
    print("PointNet test passed!")

def test_local_decoder():
    model = LocalDecoder(input_dim=1024+3, hidden_dims=[256, 128], output_dim=1)
    features = torch.randn(4, 1024)
    query_points = torch.randn(4, 100, 3)
    output = model(features, query_points)
    assert output.shape == (4, 100, 1)
    print("LocalDecoder test passed!")

if __name__ == "__main__":
    test_pointnet()
    test_local_decoder()
EOF

cat > PointSurfaceReconstruction/tests/test_utils.py << 'EOF'
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.metrics import chamfer_distance, sdf_accuracy

def test_chamfer_distance():
    pred_points = torch.randn(4, 100, 3)
    gt_points = torch.randn(4, 100, 3)
    distance = chamfer_distance(pred_points, gt_points)
    assert distance >= 0
    print("Chamfer distance test passed!")

def test_sdf_accuracy():
    pred_sdf = torch.randn(4, 100, 1)
    gt_sdf = torch.randn(4, 100, 1)
    accuracy = sdf_accuracy(pred_sdf, gt_sdf)
    assert 0 <= accuracy <= 1
    print("SDF accuracy test passed!")

if __name__ == "__main__":
    test_chamfer_distance()
    test_sdf_accuracy()
EOF

# Создание основных файлов проекта
cat > PointSurfaceReconstruction/requirements.txt << 'EOF'
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
trimesh>=3.9.0
PyYAML>=5.4.0
tqdm>=4.60.0
open3d>=0.12.0
EOF

cat > PointSurfaceReconstruction/setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="point-surface-reconstruction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "trimesh>=3.9.0",
        "PyYAML>=5.4.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Point Cloud Surface Reconstruction using Deep Learning",
    python_requires=">=3.8",
)
EOF

cat > PointSurfaceReconstruction/run_training.py << 'EOF'
#!/usr/bin/env python3
"""
Main script to run training for point cloud surface reconstruction
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from scripts.train import main

if __name__ == "__main__":
    print("Starting Point Cloud Surface Reconstruction Training...")
    main()
EOF

# Делаем скрипты исполняемыми
chmod +x PointSurfaceReconstruction/scripts/train.py
chmod +x PointSurfaceReconstruction/scripts/evaluate.py
chmod +x PointSurfaceReconstruction/scripts/preprocess_data.py
chmod +x PointSurfaceReconstruction/run_training.py

echo "Структура проекта PointSurfaceReconstruction успешно создана!"
echo "Для установки зависимостей: cd PointSurfaceReconstruction && pip install -r requirements.txt"
echo "Для запуска тестов: cd PointSurfaceReconstruction && python -m pytest tests/"
echo "Для начала обучения: cd PointSurfaceReconstruction && python run_training.py"