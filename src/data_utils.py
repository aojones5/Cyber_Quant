import os
import pickle
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader


class EmailDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(
    data_path="data/preprocessed_emails.pkl",
    max_features=5000,
    test_size=0.2,
    random_state=42,
    batch_size=32
):
    """
    Loads the preprocessed phishing email dataset, applies TF-IDF vectorization,
    splits into train/test sets, and returns PyTorch dataloaders.
    """

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Works whether pickle is a DataFrame or dict-like object
    X_raw = np.array(data["processed_text"])
    y = np.array(data["label"])

    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(X_raw).toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = EmailDataset(X_train, y_train)
    test_dataset = EmailDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "input_size": X_train.shape[1],
        "vectorizer": vectorizer,
        "X_test": X_test,
        "y_test": y_test
    }