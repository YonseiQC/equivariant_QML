import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 1) Hyperparameters
points       = 8
coords       = 3
dim_per_part = points * coords     
hidden_dim   = 48                   
output_dim   = 10                   
num_classes  = 10
batch_size   = 32
lr           = 1e-3
num_epochs   = 40
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Load & reshape data
# data     = np.load(f'dataset_{points}.npz')
data     = np.load(f'modelnet40_10classes_{points}_1.npz')
train_x  = data['train_dataset_x']  
train_y  = data['train_dataset_y']  
test_x   = data['test_dataset_x']  
test_y   = data['test_dataset_y']  

train_x = train_x.reshape(-1, points, coords).astype(np.float32)
test_x  = test_x .reshape(-1, points, coords).astype(np.float32)

# 3) DataLoaders
train_ds     = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
test_ds      = TensorDataset(torch.from_numpy(test_x),  torch.from_numpy(test_y))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

class TwoPartNet(nn.Module):
    def __init__(self, dim_part, hidden_dim, out_dim):
        super().__init__()
        self.fc1    = nn.Linear(dim_part,    hidden_dim)  
        self.fc2    = nn.Linear(hidden_dim,  hidden_dim)    
        self.fc3    = nn.Linear(hidden_dim,    hidden_dim)    
        self.fc4    = nn.Linear(hidden_dim,    hidden_dim) 
        self.fc5    = nn.Linear(hidden_dim,  hidden_dim)    
        self.fc6    = nn.Linear(hidden_dim,    hidden_dim)    
        self.fc7    = nn.Linear(hidden_dim,    hidden_dim)   
        self.fc8    = nn.Linear(hidden_dim,    hidden_dim)
        self.fc9    = nn.Linear(hidden_dim,  hidden_dim)    
        self.fc10    = nn.Linear(hidden_dim,    hidden_dim)    
        self.fc11    = nn.Linear(hidden_dim,    hidden_dim)
        self.fc12    = nn.Linear(hidden_dim,  hidden_dim)    
        self.fc13    = nn.Linear(hidden_dim,    hidden_dim)    
        self.fc14    = nn.Linear(hidden_dim,    hidden_dim)
        self.relu   = nn.ReLU(inplace=True)
        # 최종 분류기
        self.fc_out = nn.Linear(hidden_dim, out_dim)  

    def forward(self, x):
        # x shape: (batch, points, coords)
        # Flatten each part to (batch, dim_part)
        part1 = x[:,:].view(x.size(0), -1)

        # First "circuit" on part1 through 14 layers
        h = self.relu(self.fc1(part1))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        h = self.relu(self.fc5(h))
        h = self.relu(self.fc6(h))
        h = self.relu(self.fc7(h))
        h = self.relu(self.fc8(h))
        h = self.relu(self.fc9(h))
        h = self.relu(self.fc10(h))
        h = self.relu(self.fc11(h))
        h = self.relu(self.fc12(h))
        h = self.relu(self.fc13(h))
        h = self.relu(self.fc14(h))

        # Final output logits
        return self.fc_out(h)

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    """Plot confusion matrix using seaborn"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {i}' for i in range(num_classes)],
                yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

model     = TwoPartNet(dim_per_part, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 5) Train → Test loop, recording losses & accuracy
train_loss_lst = []
test_loss_lst  = []
succed_lst   = []  # test accuracy list (renamed to match second code)
class_accuracies_history = []  # track class-wise accuracies over epochs

for epoch in range(1, num_epochs+1):
    # --- Train ---
    model.train()
    running_train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * xb.size(0)
    avg_train_loss = running_train_loss / len(train_loader.dataset)
    train_loss_lst.append(avg_train_loss)

    # --- Test (compute loss & accuracy) ---
    model.eval()
    running_test_loss = 0.0
    correct = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = criterion(logits, yb)
            running_test_loss += loss.item() * xb.size(0)
            
            predictions = logits.argmax(1)
            correct += (predictions == yb).sum().item()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
    
    avg_test_loss = running_test_loss / len(test_loader.dataset)
    test_loss_lst.append(avg_test_loss)
    test_acc = correct / len(test_loader.dataset)
    succed_lst.append(test_acc)
    
    # Calculate class-wise accuracies
    class_accuracies = []
    for class_idx in range(num_classes):
        class_mask = np.array(all_targets) == class_idx
        if np.sum(class_mask) > 0:
            class_correct = np.sum((np.array(all_predictions)[class_mask] == class_idx))
            class_acc = class_correct / np.sum(class_mask)
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    class_accuracies_history.append(class_accuracies)

    # 출력
    if epoch == 1 or epoch % 10 == 0:
        print(f"[Epoch {epoch:02d}] "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Test Loss: {avg_test_loss:.4f}, "
              f"Test Acc: {test_acc:.2f}%")

# Final evaluation for detailed results
model.eval()
final_predictions = []
final_targets = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        predictions = logits.argmax(1)
        final_predictions.extend(predictions.cpu().numpy())
        final_targets.extend(yb.cpu().numpy())

# Calculate final metrics
final_overall_acc = np.mean(np.array(final_predictions) == np.array(final_targets))
final_cm = confusion_matrix(final_targets, final_predictions)

# Calculate final class accuracies
final_class_acc = []
for class_idx in range(num_classes):
    class_mask = np.array(final_targets) == class_idx
    if np.sum(class_mask) > 0:
        class_correct = np.sum((np.array(final_predictions)[class_mask] == class_idx))
        class_acc = class_correct / np.sum(class_mask)
        final_class_acc.append(class_acc)
    else:
        final_class_acc.append(0.0)

# Print final results
print("\n=== Final Results ===")
print(f"Final Overall Accuracy: {final_overall_acc:.4f}")
print("Final Class Accuracies:")
for i, acc in enumerate(final_class_acc):
    print(f"  Class {i}: {acc:.4f}")

print("\n=== Confusion Matrix ===")
print(final_cm)

plot_confusion_matrix(final_cm, title='Final Confusion Matrix')

# Create comprehensive plot with 4 subplots
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(test_loss_lst, label='Test Loss', marker='o', linestyle='-', color='b')
plt.plot(train_loss_lst, label='Training Loss', marker='o', linestyle='-', color='r')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(succed_lst, label='Test Accuracy', marker='o', linestyle='-', color='y')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
class_accuracies_array = np.array(class_accuracies_history)
for i in range(num_classes):
    plt.plot(class_accuracies_array[:, i], label=f'Class {i}', marker='o', linestyle='-')
plt.title('Class-wise Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.bar(range(num_classes), final_class_acc, color='skyblue', edgecolor='black')
plt.title('Final Class-wise Accuracy')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.xticks(range(num_classes))
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== Classification Report ===")
print(classification_report(np.array(final_targets),
                           np.array(final_predictions),
                           target_names=[f'Class {i}' for i in range(num_classes)]))
