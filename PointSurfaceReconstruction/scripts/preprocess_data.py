#!/usr/bin/env python3
"""
Скрипт для предобработки 3D данных
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import open3d as o3d
from tqdm import tqdm


def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def process_mesh_file(input_path: Path, output_dir: Path, num_points: int):
    """Обработка одного меш файла"""
    logger = logging.getLogger(__name__)
    
    try:
        # Загрузка меша
        mesh = o3d.io.read_triangle_mesh(str(input_path))
        
        if len(mesh.vertices) == 0:
            logger.warning(f"Empty mesh: {input_path}")
            return False
        
        # Нормализация меша
        mesh = normalize_mesh(mesh)
        
        # Семплирование точек
        pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
        
        # Сохранение точек
        output_path = output_dir / f"{input_path.stem}.ply"
        o3d.io.write_point_cloud(str(output_path), pcd)
        
        logger.debug(f"Processed: {input_path.name} -> {output_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return False


def normalize_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Нормализация меша к единичному bounding box"""
    # Центрирование
    vertices = np.asarray(mesh.vertices)
    center = np.mean(vertices, axis=0)
    mesh.vertices = o3d.utility.Vector3dVector(vertices - center)
    
    # Масштабирование
    vertices = np.asarray(mesh.vertices)
    max_extent = np.max(np.ptp(vertices, axis=0))
    if max_extent > 0:
        scale = 1.0 / max_extent
        mesh.vertices = o3d.utility.Vector3dVector(vertices * scale)
    
    return mesh


def main():
    """Основная функция предобработки"""
    parser = argparse.ArgumentParser(description='Preprocess 3D mesh data for training')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with mesh files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--num_points', type=int, default=5000, help='Number of points to sample per mesh')
    parser.add_argument('--formats', nargs='+', default=['obj', 'ply', 'stl'], help='Mesh file formats to process')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Поиск файлов мешей
    mesh_files = []
    for format in args.formats:
        mesh_files.extend(input_dir.glob(f'*.{format}'))
        mesh_files.extend(input_dir.glob(f'*.{format.upper()}'))
    
    if not mesh_files:
        logger.error(f"No mesh files found in {input_dir} with formats {args.formats}")
        return
    
    logger.info(f"Found {len(mesh_files)} mesh files")
    
    # Обработка файлов
    successful = 0
    for mesh_file in tqdm(mesh_files, desc="Processing meshes"):
        if process_mesh_file(mesh_file, output_dir, args.num_points):
            successful += 1
    
    logger.info(f"Successfully processed {successful}/{len(mesh_files)} files")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()