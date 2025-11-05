"""
Configuration file for pytest
"""

import pytest
import torch
import numpy as np
import random
import os


def pytest_configure(config):
    """Configuration hook for pytest"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Enable deterministic algorithms if CUDA is available
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print("PyTest configuration completed")


@pytest.fixture(autouse=True)
def set_torch_seed():
    """Automatically set random seeds before each test"""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def device():
    """Fixture to get available device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def ensure_cleanup():
    """Fixture to ensure proper cleanup after tests"""
    # Store initial state if needed
    initial_cuda_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    yield
    
    # Cleanup after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Check for memory leaks (basic check)
        final_memory = torch.cuda.memory_allocated()
        assert final_memory <= initial_cuda_memory * 2, "Possible memory leak detected"