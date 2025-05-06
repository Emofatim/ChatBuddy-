import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from preprocess import load_dataset
import joblib  # for saving the scaler

# Load and preprocess data
X, y = load_dataset("./data")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler for use in prediction
joblib.dump(scaler, "scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

# Define model
class EmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(40, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 7)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)

model = EmotionClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save model weights
torch.save(model.state_dict(), "emotion_model.pth")

# Save the model class structure
# Optional: you can pickle the full model if needed
# torch.save(model, "emotion_model_full.pth")

# Save the scaler for use in prediction
joblib.dump(scaler, "scaler.pkl")