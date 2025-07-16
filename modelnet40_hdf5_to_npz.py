#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete ModelNet40 HDF5 to NPZ converter
Handles all ModelNet40 HDF5 file structures and creates unified NPZ
"""

import h5py
import numpy as np
import os
import glob
from collections import defaultdict


def discover_modelnet40_files(data_dir):
    """
    Discover all ModelNet40 HDF5 files in directory
    
    Returns:
        train_files: List of training H5 files
        test_files: List of test H5 files
    """
    print(f"🔍 Discovering ModelNet40 files in: {data_dir}")
    
    # Common ModelNet40 file patterns
    train_patterns = [
        "ply_data_train*.h5",
        "*train*.h5", 
        "modelnet40_train*.h5",
        "train*.h5"
    ]
    
    test_patterns = [
        "ply_data_test*.h5",
        "*test*.h5",
        "modelnet40_test*.h5", 
        "test*.h5"
    ]
    
    train_files = []
    test_files = []
    
    # Find train files
    for pattern in train_patterns:
        found = glob.glob(os.path.join(data_dir, pattern))
        train_files.extend(found)
    
    # Find test files  
    for pattern in test_patterns:
        found = glob.glob(os.path.join(data_dir, pattern))
        test_files.extend(found)
    
    # Remove duplicates and sort
    train_files = sorted(list(set(train_files)))
    test_files = sorted(list(set(test_files)))
    
    print(f"📂 Found {len(train_files)} train files:")
    for f in train_files:
        print(f"   - {os.path.basename(f)}")
    
    print(f"📂 Found {len(test_files)} test files:")
    for f in test_files:
        print(f"   - {os.path.basename(f)}")
    
    if not train_files and not test_files:
        print("❌ No ModelNet40 files found!")
        print("Expected file patterns:")
        print("  - ply_data_train0.h5, ply_data_train1.h5, ...")
        print("  - ply_data_test0.h5, ply_data_test1.h5, ...")
        
    return train_files, test_files


def analyze_h5_structure(h5_file_path):
    """
    Analyze single H5 file structure
    """
    print(f"\n🔬 Analyzing: {os.path.basename(h5_file_path)}")
    
    with h5py.File(h5_file_path, 'r') as f:
        structure = {}
        
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                dataset = f[key]
                structure[key] = {
                    'shape': dataset.shape,
                    'dtype': dataset.dtype,
                    'size': dataset.size
                }
                print(f"   {key}: {dataset.shape}, {dataset.dtype}")
        
        return structure


def load_h5_data(h5_file_path):
    """
    Load data from single H5 file with robust key detection
    """
    with h5py.File(h5_file_path, 'r') as f:
        data_dict = {}
        
        # Standard ModelNet40 keys
        key_mapping = {
            'data': ['data', 'point_cloud', 'points', 'coordinates'],
            'label': ['label', 'labels', 'class', 'target', 'y'],
            'normal': ['normal', 'normals', 'norm'],
            'pid': ['pid', 'point_id', 'id']
        }
        
        # Find and load available data
        for standard_key, possible_keys in key_mapping.items():
            for key in possible_keys:
                if key in f.keys():
                    data_dict[standard_key] = f[key][:]
                    print(f"     ✓ {key} -> {standard_key}: {f[key].shape}")
                    break
        
        # Load any other datasets not in standard mapping
        for key in f.keys():
            if key not in [k for keys in key_mapping.values() for k in keys]:
                if isinstance(f[key], h5py.Dataset):
                    data_dict[key] = f[key][:]
                    print(f"     ? {key}: {f[key].shape}")
        
        return data_dict


def combine_h5_files(file_list, file_type="train"):
    """
    Combine multiple H5 files into unified arrays
    """
    if not file_list:
        print(f"⚠️  No {file_type} files to process")
        return None
    
    print(f"\n📦 Combining {len(file_list)} {file_type} files...")
    
    combined_data = defaultdict(list)
    total_samples = 0
    
    for i, h5_file in enumerate(file_list):
        print(f"   [{i+1}/{len(file_list)}] Processing {os.path.basename(h5_file)}")
        
        try:
            file_data = load_h5_data(h5_file)
            
            # Track sample counts
            if 'data' in file_data:
                samples_in_file = file_data['data'].shape[0]
                total_samples += samples_in_file
                print(f"       {samples_in_file} samples")
            
            # Combine all available keys
            for key, value in file_data.items():
                combined_data[key].append(value)
                
        except Exception as e:
            print(f"       ❌ Error: {e}")
            continue
    
    # Concatenate arrays
    final_data = {}
    for key, value_list in combined_data.items():
        if value_list:
            try:
                final_data[key] = np.concatenate(value_list, axis=0)
                print(f"   ✓ {key}: {final_data[key].shape}")
            except Exception as e:
                print(f"   ❌ Failed to combine {key}: {e}")
    
    print(f"   📊 Total {file_type} samples: {total_samples}")
    return final_data


def validate_modelnet40_data(train_data, test_data):
    """
    Validate combined ModelNet40 data
    """
    print(f"\n🔍 Validating ModelNet40 data...")
    
    # Check required keys
    required_keys = ['data', 'label']
    for key in required_keys:
        if key not in train_data:
            raise ValueError(f"Missing required key '{key}' in train data")
        if key not in test_data:
            raise ValueError(f"Missing required key '{key}' in test data")
    
    # Validate shapes
    train_samples = train_data['data'].shape[0]
    test_samples = test_data['data'].shape[0]
    
    print(f"   Train samples: {train_samples}")
    print(f"   Test samples: {test_samples}")
    
    # Check ModelNet40 standard numbers (with tolerance)
    expected_train = 9843
    expected_test = 2468
    
    if abs(train_samples - expected_train) > 100:
        print(f"   ⚠️  Train samples ({train_samples}) differ from standard ({expected_train})")
    
    if abs(test_samples - expected_test) > 100:
        print(f"   ⚠️  Test samples ({test_samples}) differ from standard ({expected_test})")
    
    # Validate data dimensions
    if len(train_data['data'].shape) != 3:
        raise ValueError(f"Expected 3D train data, got shape {train_data['data'].shape}")
    
    if train_data['data'].shape[2] != 3:
        raise ValueError(f"Expected 3D coordinates, got {train_data['data'].shape[2]} dimensions")
    
    # Validate labels
    train_classes = len(np.unique(train_data['label']))
    test_classes = len(np.unique(test_data['label']))
    
    print(f"   Train classes: {train_classes}")
    print(f"   Test classes: {test_classes}")
    
    if train_classes != 40 or test_classes != 40:
        print(f"   ⚠️  Expected 40 classes, got train={train_classes}, test={test_classes}")
    
    # Check label ranges
    train_label_range = [train_data['label'].min(), train_data['label'].max()]
    test_label_range = [test_data['label'].min(), test_data['label'].max()]
    
    print(f"   Train label range: {train_label_range}")
    print(f"   Test label range: {test_label_range}")
    
    # Check coordinate ranges
    train_coord_range = [train_data['data'].min(), train_data['data'].max()]
    test_coord_range = [test_data['data'].min(), test_data['data'].max()]
    
    print(f"   Train coord range: [{train_coord_range[0]:.3f}, {train_coord_range[1]:.3f}]")
    print(f"   Test coord range: [{test_coord_range[0]:.3f}, {test_coord_range[1]:.3f}]")
    
    print("   ✅ Data validation completed!")


def create_modelnet40_npz(data_dir, output_path="modelnet40.npz", format_style="train_test"):
    """
    Complete ModelNet40 H5 to NPZ conversion
    
    Args:
        data_dir: Directory containing H5 files
        output_path: Output NPZ file path
        format_style: 'train_test', 'X_y', or 'combined'
    """
    print("🚀 ModelNet40 HDF5 → NPZ Conversion Started")
    print("=" * 60)
    
    # Discover files
    train_files, test_files = discover_modelnet40_files(data_dir)
    
    if not train_files and not test_files:
        raise FileNotFoundError("No ModelNet40 H5 files found!")
    
    # Analyze first file structure
    sample_file = train_files[0] if train_files else test_files[0]
    analyze_h5_structure(sample_file)
    
    # Combine files
    train_data = combine_h5_files(train_files, "train") if train_files else None
    test_data = combine_h5_files(test_files, "test") if test_files else None
    
    if train_data is None or test_data is None:
        raise ValueError("Failed to load train or test data")
    
    # Validate combined data
    validate_modelnet40_data(train_data, test_data)
    
    # Prepare save dictionary based on format style
    save_dict = {}
    
    if format_style == "train_test":
        # Format: train_data, train_label, test_data, test_label
        save_dict.update({
            'train_data': train_data['data'],
            'train_label': train_data['label'],
            'test_data': test_data['data'], 
            'test_label': test_data['label']
        })
        
    elif format_style == "X_y":
        # Format: X_train, y_train, X_test, y_test
        save_dict.update({
            'X_train': train_data['data'],
            'y_train': train_data['label'],
            'X_test': test_data['data'],
            'y_test': test_data['label']
        })
        
    elif format_style == "combined":
        # Format: data, label (combined)
        all_data = np.concatenate([train_data['data'], test_data['data']], axis=0)
        all_label = np.concatenate([train_data['label'], test_data['label']], axis=0)
        save_dict.update({
            'data': all_data,
            'label': all_label
        })
        
    # Add optional data if available
    optional_keys = ['normal', 'pid']
    for key in optional_keys:
        if key in train_data and key in test_data:
            if format_style == "train_test":
                save_dict[f'train_{key}'] = train_data[key]
                save_dict[f'test_{key}'] = test_data[key]
            elif format_style == "X_y":
                save_dict[f'{key}_train'] = train_data[key]
                save_dict[f'{key}_test'] = test_data[key]
            elif format_style == "combined":
                combined_optional = np.concatenate([train_data[key], test_data[key]], axis=0)
                save_dict[key] = combined_optional
            
            print(f"   ✓ Added optional data: {key}")
    
    # Add metadata
    save_dict.update({
        'num_classes': 40,
        'num_train_samples': train_data['data'].shape[0],
        'num_test_samples': test_data['data'].shape[0],
        'points_per_sample': train_data['data'].shape[1],
        'coordinate_dims': train_data['data'].shape[2],
        'format_style': format_style,
        'source': 'ModelNet40'
    })
    
    # Save NPZ file
    print(f"\n💾 Saving to: {output_path}")
    np.savez_compressed(output_path, **save_dict)
    
    # Final summary
    file_size = os.path.getsize(output_path) / (1024**2)
    print(f"\n🎉 Conversion completed successfully!")
    print(f"   Output file: {output_path}")
    print(f"   File size: {file_size:.1f} MB")
    print(f"   Format: {format_style}")
    print(f"   Keys saved: {list(save_dict.keys())[:8]}...")  # Show first 8 keys
    
    return output_path


def test_npz_file(npz_path):
    """
    Test the created NPZ file
    """
    print(f"\n🧪 Testing NPZ file: {npz_path}")
    
    with np.load(npz_path) as data:
        print("Available keys:")
        for key in data.files:
            if key in data:
                arr = data[key]
                if hasattr(arr, 'shape'):
                    print(f"   {key}: {arr.shape}, {arr.dtype}")
                else:
                    print(f"   {key}: {arr}")


# Usage examples
if __name__ == "__main__":
    # 실제 폴더 경로
    data_directory = "/Users/dang-geun/Desktop/modelnet40_ply_hdf5_2048"
    
    print(f"🎯 Target directory: {data_directory}")
    
    if not os.path.exists(data_directory):
        print(f"❌ Directory not found: {data_directory}")
        print("Please check if the folder exists and the path is correct")
        exit(1)
    
    # 폴더 내용 확인
    files_in_dir = os.listdir(data_directory)
    h5_files = [f for f in files_in_dir if f.endswith('.h5')]
    print(f"📁 Found {len(h5_files)} H5 files in directory")
    
    if not h5_files:
        print("❌ No .h5 files found in the directory!")
        print("Files in directory:", files_in_dir[:10])  # Show first 10 files
        exit(1)
    
    try:
        print(f"\n{'='*60}")
        print("🚀 Creating ModelNet40.npz for your quantum ML code...")
        
        # Create NPZ in train_test format (compatible with your code)
        output_file = create_modelnet40_npz(
            data_directory, 
            "modelnet40.npz",  # This will be created in current directory
            "train_test"       # Format compatible with your load_modelnet40_npz()
        )
        
        # Test the created file
        test_npz_file(output_file)
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Created: {output_file}")
        print(f"✅ Ready for your quantum ML code!")
        print(f"\nNow you can run your makepoint(4, 2) code! 🚀")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

# Quick execution function
def quick_convert():
    """Quick conversion for the specific directory"""
    data_dir = "/Users/dang-geun/Desktop/modelnet40_ply_hdf5_2048"
    output_path = "modelnet40.npz"
    
    print("🚀 Quick ModelNet40 conversion starting...")
    
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return None
    
    try:
        result = create_modelnet40_npz(data_dir, output_path, "train_test")
        print(f"✅ Quick conversion completed: {result}")
        return result
    except Exception as e:
        print(f"❌ Quick conversion failed: {e}")
        return None

# Uncomment the line below to run quick conversion
# quick_convert()