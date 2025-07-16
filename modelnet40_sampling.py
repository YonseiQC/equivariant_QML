#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modified data.py for Quantum ML - using ModelNet40.npz with label filtering (0-9)
Test set fixed to 256 samples, rest for training
"""

import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from scipy.stats import special_ortho_group


def load_modelnet40_npz():
    """Load ModelNet40 data from npz file and filter labels 0-9 with fixed test size of 256"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(BASE_DIR, 'modelnet40.npz')
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"ModelNet40.npz not found at: {npz_path}")
    
    print(f"Loading ModelNet40 data from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    # Print available keys for debugging
    print(f"Available keys in npz: {list(data.keys())}")
    
    # Try different possible key naming conventions
    try:
        if 'train_data' in data.keys():
            # Convention 1: train_data, train_label, test_data, test_label
            all_data = np.concatenate([data['train_data'], data['test_data']])
            all_label = np.concatenate([data['train_label'], data['test_label']])
            
        elif 'x_train' in data.keys():
            # Convention 2: x_train, y_train, x_test, y_test (lowercase)
            all_data = np.concatenate([data['x_train'], data['x_test']])
            all_label = np.concatenate([data['y_train'], data['y_test']])
            
        elif 'data' in data.keys() and 'label' in data.keys():
            # Convention 3: data, label (combined)
            all_data = data['data']
            all_label = data['label']
        else:
            raise KeyError(f"Unknown npz format. Available keys: {list(data.keys())}")
        
        print(f"Total data: {all_data.shape}, Total labels: {all_label.shape}")
        print(f"Original label range: {all_label.min()} - {all_label.max()}")
        
        if len(all_label.shape) > 1:
            all_label = all_label.flatten()
            print(f"Flattened labels to: {all_label.shape}")
        

        filter_mask = (all_label == 1) | (all_label == 2) | (all_label == 8) | (all_label == 12) | (all_label == 14) | (all_label == 22) | (all_label == 23) | (all_label == 30) | (all_label == 33) | (all_label == 35)
        print(f"Filter mask shape: {filter_mask.shape}, True count: {filter_mask.sum()}")
        
        filtered_data = all_data[filter_mask]
        filtered_label = all_label[filter_mask]
        
        print(f"After filtering - data: {filtered_data.shape}, labels: {filtered_label.shape}")
        print(f"Filtered unique labels: {np.unique(filtered_label)}")
        

        train_data = filtered_data[:-256]
        train_label = filtered_label[:-256] 
        test_data = filtered_data[-256:]
        test_label = filtered_label[-256:]
        
        print(f"✅ Final split:")
        print(f"Train: {train_data.shape}, Test: {test_data.shape}")
        print(f"Train unique labels: {np.unique(train_label)}")
        print(f"Test unique labels: {np.unique(test_label)}")
        
        return train_data, train_label, test_data, test_label
        
    except Exception as e:
        print(f"❌ Error loading npz data: {e}")
        print("Expected npz format:")
        print("  Option 1: train_data, train_label, test_data, test_label")
        print("  Option 2: x_train, y_train, x_test, y_test") 
        print("  Option 3: data, label (combined)")
        raise


class ModelNet40(Dataset):
    def __init__(self, num_qubit, num_reupload, partition='train'):
        # Load data from npz file (already filtered for labels 0-9)
        train_data, train_label, test_data, test_label = load_modelnet40_npz()
        
        if partition == 'train':
            self.data = train_data
            self.label = train_label
        else:
            self.data = test_data
            self.label = test_label
        
        # Calculate num_points from quantum parameters
        self.num_points = num_reupload * num_qubit
        self.num_qubit = num_qubit
        self.num_reupload = num_reupload
        self.partition = partition
        
        print(f"Final {partition} dataset: {len(self.data)} samples")
        print(f"Using {self.num_points} points per sample (num_qubit={num_qubit}, num_reupload={num_reupload})")
        print(f"Unique labels in {partition} set: {np.unique(self.label)}")

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


def makepoint(num_qubit, num_reupload, rotation):
    """
    Create quantum-optimized point cloud datasets for ModelNet40 with label filtering (0-9)
    Test set fixed to 256 samples, rest for training
    
    Args:
        num_qubit: Number of qubits
        num_reupload: Number of data reuploads
    
    Returns:
        train_dataset_x, train_dataset_y, test_dataset_x, test_dataset_y
    """
    print(f"Creating quantum-optimized ModelNet40 datasets with label filtering (0-9)...")
    print(f"Test set fixed to 256 samples, rest for training")
    print(f"Parameters: num_qubit={num_qubit}, num_reupload={num_reupload}")
    print(f"Points per sample: {num_qubit * num_reupload}")
    
    train_dataset = ModelNet40(num_qubit, num_reupload, 'train')
    test_dataset = ModelNet40(num_qubit, num_reupload, 'test')

    print("Processing data and collecting labels...")
    train_dataset_x = []
    train_dataset_y = []
    test_dataset_x = []
    test_dataset_y = []
    all_labels = []

    # Train data processing
    for i in range(len(train_dataset)):
        pointcloud, label = train_dataset[i]
        train_dataset_x.append(pointcloud)
        train_dataset_y.append(label)
        all_labels.append(label)

    # Test data processing  
    for i in range(len(test_dataset)):
        pointcloud, label = test_dataset[i]
        test_dataset_x.append(pointcloud)
        test_dataset_y.append(label)
        all_labels.append(label)

    # LabelEncoder 학습 및 변환
    encoder = LabelEncoder()
    encoder.fit(all_labels)

    train_dataset_x = np.array(train_dataset_x)
    train_dataset_y = np.array(encoder.transform(train_dataset_y))
    test_dataset_x = np.array(test_dataset_x)
    test_dataset_y = np.array(encoder.transform(test_dataset_y))
    
    print(f"\nLabel verification:")
    print(f"Train labels range: {train_dataset_y.min()} - {train_dataset_y.max()}")
    print(f"Test labels range: {test_dataset_y.min()} - {test_dataset_y.max()}")
    print(f"Train unique labels: {np.unique(train_dataset_y)}")
    print(f"Test unique labels: {np.unique(test_dataset_y)}")
    
    if rotation == 0:
        filename = f"modelnet40_10classes_{num_qubit}_{num_reupload}.npz"

    if rotation == 1:
        filename = f"modelnet40_10classes_{num_qubit}_{num_reupload}_rotation.npz"

        rot = special_ortho_group.rvs(3)
        train_dataset_x = train_dataset_x @ rot
        test_dataset_x = test_dataset_x @ rot

    np.savez_compressed(filename,
                      train_dataset_x=train_dataset_x,
                      train_dataset_y=train_dataset_y, 
                      test_dataset_x=test_dataset_x,
                      test_dataset_y=test_dataset_y,
                      num_qubit=num_qubit,
                      num_reupload=num_reupload,
                      label_range="0-9",
                      num_classes=10)
    
    print(f"\nSaved all datasets to {filename}")
    print(f"Train data shape: {train_dataset_x.shape}")
    print(f"Train label shape: {train_dataset_y.shape}")
    print(f"Test data shape: {test_dataset_x.shape}")
    print(f"Test label shape: {test_dataset_y.shape}")
    print(f"✅ Succeed! Dataset created with 10 classes (labels 0-9), test fixed to 256 samples")

    return train_dataset_x, train_dataset_y, test_dataset_x, test_dataset_y

makepoint(8, 1, 1)