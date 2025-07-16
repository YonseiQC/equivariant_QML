import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import random

# GPU 사용 가능한지 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def random_sample_points(points, num_sample):
    """
    원래 PointNet 방식: 무작위 점 샘플링
    Args:
        points: (N, 3) numpy array
        num_sample: 샘플링할 점의 개수
    Returns:
        sampled_points: (num_sample, 3) numpy array
    """
    N = points.shape[0]
    indices = np.random.choice(N, num_sample, replace=False)
    return points[indices]

def calculate_class_accuracies(y_true, y_pred, num_classes):
    """각 클래스별 정확도 계산"""
    class_accuracies = []
    for i in range(num_classes):
        class_mask = (y_true == i)
        if torch.sum(class_mask) > 0:
            class_acc = torch.sum((y_pred == i) & class_mask).float() / torch.sum(class_mask).float()
            class_accuracies.append(class_acc.item())
        else:
            class_accuracies.append(0.0)
    return class_accuracies

def calculate_final_metrics(y_true, y_pred, num_classes):
    """최종 메트릭 계산"""
    # Convert to numpy if tensor
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Overall accuracy
    overall_acc = accuracy_score(y_true, y_pred)
    
    # Class accuracies
    class_accuracies = []
    for i in range(num_classes):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            class_acc = np.sum((y_pred == i) & class_mask) / np.sum(class_mask)
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    return cm, class_accuracies, overall_acc

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    """Confusion matrix 시각화"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {i}' for i in range(len(cm))],
                yticklabels=[f'Class {i}' for i in range(len(cm))])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    """
    원래 PointNet 방식: Point cloud를 단위구에 맞게 정규화
    Args:
        points: (N, 3) numpy array
    Returns:
        normalized_points: (N, 3) numpy array
    """
    # 중심을 원점으로 이동
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # 최대 거리로 스케일링
    m = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / m
    
    return points

class PointNetDataset(Dataset):
    def __init__(self, points, labels, num_points=28, is_training=False):
        """
        원래 PointNet 방식의 데이터셋 로더
        Args:
            points: (N, original_num_points, 3) numpy array
            labels: (N,) numpy array
            num_points: 샘플링할 점의 개수 (28개)
            is_training: 학습용인지 여부 (데이터 증강 적용)
        """
        self.points = points
        self.labels = labels
        
        self.num_points = num_points
        self.is_training = is_training
        
        print(f"Loaded dataset: {self.points.shape[0]} samples")
        print(f"Original point cloud shape: {self.points.shape[1:]}") 
        print(f"Target points: {num_points}")
        print(f"Number of classes: {len(np.unique(self.labels))}")
        
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, idx):
        points = self.points[idx].astype(np.float32)  # (original_num_points, 3)
        label = self.labels[idx].astype(np.int64)
        
        # 원래 PointNet 방식: uniform random sampling
        if points.shape[0] != self.num_points:
            points = random_sample_points(points, self.num_points)
        
        # 원래 PointNet 방식: training 시에만 간단한 데이터 증강
        if self.is_training:
            # Random rotation around z-axis
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0], 
                [0, 0, 1]
            ])
            points = points @ rotation_matrix.T
            
            # Small jitter
            points += np.random.normal(0, 0.02, points.shape)
            
        # 명시적으로 float32로 변환
        return torch.from_numpy(points).float(), torch.from_numpy(np.array(label)).long()

class TNet(nn.Module):
    """표준 PointNet T-Net (Transformation Network)"""
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        
        # 표준 PointNet T-Net 구조
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Shared MLPs
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        
        # MLPs
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Add identity matrix
        identity = torch.eye(self.k, device=x.device).view(1, self.k*self.k).repeat(batch_size, 1)
        x = x + identity
        x = x.view(batch_size, self.k, self.k)
        
        return x

class PointNet(nn.Module):
    """표준 PointNet Classification Network"""
    def __init__(self, num_classes=10):
        super(PointNet, self).__init__()
        
        # T-Nets
        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=64)
        
        # Shared MLPs (표준 PointNet 구조)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        
        # Classification head (표준 PointNet 구조)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(p=0.3)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        batch_size = x.size(0)
        n_points = x.size(2)
        
        # Input transformation
        trans = self.input_transform(x)  # (B, 3, 3)
        x = x.transpose(2, 1)  # (B, n_points, 3)
        x = torch.bmm(x, trans)  # Apply transformation
        x = x.transpose(2, 1)  # (B, 3, n_points)
        
        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Feature transformation
        trans_feat = self.feature_transform(x)  # (B, 64, 64)
        x = x.transpose(2, 1)  # (B, n_points, 64)
        x = torch.bmm(x, trans_feat)  # Apply feature transformation
        x = x.transpose(2, 1)  # (B, 64, n_points)
        
        # Continue shared MLP
        pointfeat = x  # Save point features
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Global feature (max pooling)
        x = torch.max(x, 2, keepdim=True)[0]  # (B, 1024, 1)
        x = x.view(batch_size, -1)  # (B, 1024)
        
        # Classification
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    """Feature transformation regularization"""
    d = trans.size()[1]
    I = torch.eye(d, device=trans.device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

def train_model(model, train_loader, test_loader, num_classes, num_epochs, learning_rate, weight_decay):
    """원래 PointNet 방식의 학습 (validation 없이)"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_loss_lst = []
    test_loss_lst = []
    test_accuracy_lst = []
    class_accuracies_history = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)
            points = points.transpose(2, 1)  # (B, 3, 28)
            
            optimizer.zero_grad()
            
            pred, trans, trans_feat = model(points)
            loss = criterion(pred, labels)
            
            # Feature transform regularization (원래 PointNet loss)
            reg_loss = feature_transform_regularizer(trans_feat)
            loss += 0.001 * reg_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        train_loss_lst.append(train_loss)
        print(f"{train_loss:.6f}")  # 훈련 손실 출력
        
        # Test evaluation
        model.eval()
        test_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for points, labels in test_loader:
                points, labels = points.to(device), labels.to(device)
                points = points.transpose(2, 1)  # (B, 3, 28)
                
                pred, trans, trans_feat = model(points)
                loss = criterion(pred, labels)
                reg_loss = feature_transform_regularizer(trans_feat)
                loss += 0.001 * reg_loss
                
                test_loss += loss.item()
                
                _, predicted = torch.max(pred, 1)
                all_predictions.extend(predicted.cpu())
                all_labels.extend(labels.cpu())
        
        test_loss = test_loss / len(test_loader)
        test_loss_lst.append(test_loss)
        print(f"{test_loss:.6f}")  # 테스트 손실 출력
        
        # Calculate accuracies
        all_predictions = torch.tensor(all_predictions)
        all_labels = torch.tensor(all_labels)
        
        test_accuracy = torch.sum(all_predictions == all_labels).float() / len(all_labels)
        test_accuracy_lst.append(test_accuracy.item())
        
        class_accuracies = calculate_class_accuracies(all_labels, all_predictions, num_classes)
        class_accuracies_history.append(class_accuracies)
        
        print(f"{epoch}, {test_accuracy:.4f}")  # 에포크, 테스트 정확도 출력
        
        # 클래스별 정확도 출력
        for i, acc in enumerate(class_accuracies):
            print(f"Class {i}: {acc:.4f}")
        print("-" * 30)
        
        # Learning rate decay (원래 PointNet에서 사용한 간단한 방식)
        if epoch == 120:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        if epoch == 180:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
    
    # 최종 평가
    model.eval()
    final_predictions = []
    final_labels = []
    
    with torch.no_grad():
        for points, labels in test_loader:
            points, labels = points.to(device), labels.to(device)
            points = points.transpose(2, 1)
            
            pred, _, _ = model(points)
            _, predicted = torch.max(pred, 1)
            
            final_predictions.extend(predicted.cpu())
            final_labels.extend(labels.cpu())
    
    final_predictions = torch.tensor(final_predictions)
    final_labels = torch.tensor(final_labels)
    
    # 최종 메트릭 계산
    final_cm, final_class_acc, final_overall_acc = calculate_final_metrics(
        final_labels, final_predictions, num_classes)
    
    # 최종 결과 출력
    print("\n=== Final Results ===")
    print(f"Final Overall Accuracy: {final_overall_acc:.4f}")
    print("Final Class Accuracies:")
    for i, acc in enumerate(final_class_acc):
        print(f"  Class {i}: {acc:.4f}")
    
    print("\n=== Confusion Matrix ===")
    print(final_cm)
    
    # Confusion matrix 시각화
    plot_confusion_matrix(final_cm, title='Final Confusion Matrix')
    
    # 결과 시각화
    plt.figure(figsize=(12, 8))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(test_loss_lst, label='Test Loss', marker='o', linestyle='-', color='b')
    plt.plot(train_loss_lst, label='Training Loss', marker='o', linestyle='-', color='r')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()  
    plt.grid(True)
    
    # Test accuracy plot
    plt.subplot(2, 2, 2)
    plt.plot(test_accuracy_lst, label='Test Accuracy', marker='o', linestyle='-', color='y')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()  
    plt.grid(True)
    
    # Class-wise accuracy over epochs
    plt.subplot(2, 2, 3)
    class_accuracies_array = np.array(class_accuracies_history)
    for i in range(num_classes):
        plt.plot(class_accuracies_array[:, i], label=f'Class {i}', marker='o', linestyle='-')
    plt.title('Class-wise Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Final class-wise accuracy bar chart
    plt.subplot(2, 2, 4)
    plt.bar(range(num_classes), final_class_acc, color='skyblue', edgecolor='black')
    plt.title('Final Class-wise Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(range(num_classes))
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(np.array(final_labels).flatten(), 
                                np.array(final_predictions).flatten(),
                                target_names=[f'Class {i}' for i in range(num_classes)]))
    
    return train_loss_lst, test_accuracy_lst

def evaluate_model(model, test_loader):
    """최종 모델 평가 (기존 호환성을 위해 유지)"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for points, labels in test_loader:
            points, labels = points.to(device), labels.to(device)
            points = points.transpose(2, 1)  # (B, 3, 28)
            
            pred, _, _ = model(points)
            _, predicted = torch.max(pred, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy

def plot_training_history(train_losses, test_accuracies):
    """기존 호환성을 위한 간단한 시각화 (실제로는 train_model에서 더 자세한 시각화 제공)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(test_accuracies)
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

def main(num_points, num_epochs, learning_rate, batch_size, weight_decay):
    """
    Args:
        num_points: 사용할 점의 개수
        num_epochs: 학습 에포크 수
        learning_rate: 학습률
        batch_size: 배치 크기
        weight_decay: 가중치 감쇠
    """
    # 점 개수에 따른 데이터셋 파일명 생성
    data_filename = f'modelnet40_10classes_{num_points}_1.npz'
    
    print(f"Loading dataset: {data_filename}")
    print(f"Using {num_points} points")
    
    # 전체 데이터셋 로드 (이미 train/test가 분리되어 있음)
    try:
        data = np.load(data_filename)
        train_points = data['train_dataset_x'].astype(np.float32)
        train_labels = data['train_dataset_y'].astype(np.int64)
        test_points = data['test_dataset_x'].astype(np.float32)
        test_labels = data['test_dataset_y'].astype(np.int64)
    except FileNotFoundError:
        print(f"Error: 파일 '{data_filename}'를 찾을 수 없습니다.")
        print(f"파일명이 올바른지 확인해주세요.")
        return
    except KeyError as e:
        print(f"Error: npz 파일에서 키 {e}를 찾을 수 없습니다.")
        print("예상되는 키: 'train_dataset_x', 'train_dataset_y', 'test_dataset_x', 'test_dataset_y'")
        return
    
    # 데이터셋 생성 (원래 PointNet 방식)
    train_dataset = PointNetDataset(train_points, train_labels, num_points=num_points, is_training=True)
    test_dataset = PointNetDataset(test_points, test_labels, num_points=num_points, is_training=False)
    
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    print("원래 PointNet 방식: 미리 분리된 train/test 사용")
    
    # 데이터 로더
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 원래 PointNet 모델 생성
    num_classes = len(np.unique(train_labels))
    model = PointNet(num_classes=num_classes).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 샘플 데이터 확인
    sample_data = train_dataset[0]
    print(f"Sample point cloud shape: {sample_data[0].shape}")
    print(f"Sample label: {sample_data[1]}")
    
    # 원래 PointNet 방식으로 학습
    print(f"\n원래 PointNet 방식으로 학습 시작 ({num_points} points)...")
    print("- Uniform random sampling")
    print("- Simple data augmentation (rotation + jitter)")
    print("- Pre-separated train/test datasets")
    print("- Fixed learning rate schedule")
    print("- Data already normalized to unit sphere")
    
    train_losses, test_accuracies = train_model(
        model, train_loader, test_loader, num_classes,
        num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay
    )
    
    # 기존 호환성을 위한 간단한 최종 평가
    final_test_accuracy = evaluate_model(model, test_loader)
    
    print(f"\n이 결과는 'PointNet architecture with {num_points} points'의 성능입니다.")
    print("원래 PointNet 논문과 동일한 방식으로 학습됨:")
    print("- Uniform random sampling")
    print("- Simple data augmentation (z-axis rotation + jitter)")  
    print("- Pre-separated train/test datasets")
    print("- Data already normalized to unit sphere")
    
    # 최종 모델 저장
    model_save_path = f'pointnet_{num_points}points_final.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"최종 모델이 '{model_save_path}'로 저장되었습니다.")
    
    return final_test_accuracy

# 여러 점 개수로 실험하는 함수
def experiment_with_different_points(point_numbers, num_epochs, learning_rate, batch_size, weight_decay):
    """
    여러 점 개수로 실험을 진행하는 함수
    Args:
        point_numbers: 실험할 점 개수 리스트
        num_epochs: 학습 에포크 수
        learning_rate: 학습률
        batch_size: 배치 크기
        weight_decay: 가중치 감쇠
    """
    results = {}
    
    for num_points in point_numbers:
        print(f"\n{'='*50}")
        print(f"실험 시작: {num_points} points")
        print(f"{'='*50}")
        
        try:
            accuracy = main(num_points, num_epochs, learning_rate, batch_size, weight_decay)
            results[num_points] = accuracy
            print(f"{num_points} points 실험 완료: {accuracy:.4f}")
        except Exception as e:
            print(f"{num_points} points 실험 실패: {e}")
            results[num_points] = None
    
    # 결과 요약
    print(f"\n{'='*50}")
    print("실험 결과 요약")
    print(f"{'='*50}")
    for num_points, accuracy in results.items():
        if accuracy is not None:
            print(f"{num_points:3d} points: {accuracy:.4f}")
        else:
            print(f"{num_points:3d} points: 실패")
    
    return results

def result(num_points, num_epochs, learning_rate, batch_size, weight_decay):
    """결과 함수 - 하이퍼파라미터 출력 후 학습 실행"""
    print(f'num_points = {num_points}, num_epochs = {num_epochs}, learning_rate = {learning_rate}, batch_size = {batch_size}, weight_decay = {weight_decay}')
    return main(num_points, num_epochs, learning_rate, batch_size, weight_decay)

# ==================== 하이퍼파라미터 설정 ====================
num_points = 8
num_epochs = 5
learning_rate = 0.001
batch_size = 32
weight_decay = 1e-4

# 실험 실행
result(num_points, num_epochs, learning_rate, batch_size, weight_decay)

# 여러 점 개수로 실험하고 싶다면 아래 주석 해제
# point_numbers = [28, 64, 128, 256]
# experiment_with_different_points(point_numbers, num_epochs, learning_rate, batch_size, weight_decay)
