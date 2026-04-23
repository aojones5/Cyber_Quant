import torch 
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#load pre processed emails pickle files
with open('preprocessed_emails.pkl', 'rb') as f: 
    data = pickle.load(f)

#populate lists 
X_raw = np.array(data['processed_text']) # features (input data)
y = np.array(data['label'])# labels (target variables)

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features for efficiency
X = vectorizer.fit_transform(X_raw).toarray()

#split into traing and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#convert to pytorch tensors MAY NEED TO CHANGE DTYPE 
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long).view(-1)
y_test = torch.tensor(y_test, dtype=torch.long).view(-1)

# Create a PyTorch Dataset 
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    #could add __str__ method for debugging

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

# Create DataLoaders
train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


#Define Nerual Network  
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

#try and train on CPU (faster)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

#Set up Model 
input_size = X_train.shape[1]
model = NeuralNet(input_size=input_size , hidden_size=128, num_classes=2).to(device)

#define loss function
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Validation loop
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")