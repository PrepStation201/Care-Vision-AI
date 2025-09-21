import torch
from tqdm import tqdm

class ModelTrainer:
    """
    A class to handle the training of a PyTorch model.
    """
    def __init__(self, model, train_loader, criterion, optimizer, scheduler, device):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, epoch, epochs):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)

    def train(self, epochs):
        """
        Trains the model for a specified number of epochs.
        """
        train_loss = []
        for epoch in range(epochs):
            epoch_loss = self.train_epoch(epoch, epochs)
            train_loss.append(epoch_loss)
            self.scheduler.step(epoch_loss)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        return train_loss