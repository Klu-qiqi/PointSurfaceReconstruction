#!/usr/bin/env python3
"""
Главный скрипт запуска обучения
Альтернатива scripts/train.py с дополнительными функциями
"""

import sys
from pathlib import Path

# Добавляем путь к src
sys.path.append(str(Path(__file__).parent / 'src'))

from scripts.train import main

if __name__ == "__main__":
    main()