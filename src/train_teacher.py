import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)

def train_teacher(teacher, cifar_train, cifar_test, device, learning_rate, num_epochs):
    teacher = teacher.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=learning_rate)
    
    try:
        for epoch in range(num_epochs):
            teacher.train()
            running_loss = 0.0

            logger.info(f"Epoch [{epoch + 1}/{num_epochs}] started.")
            for i, (inputs, labels) in enumerate(cifar_train):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = teacher(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        
            teacher.eval()
            correct, total, test_loss = 0, 0, 0
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(cifar_test):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = teacher(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            accuracy = 100. * correct / total
            test_loss /= (i + 1) 

            logger.info(f'Epoch {epoch+1}, Training Loss: {running_loss/(i+1):.3f}, Test Loss: {test_loss:.3f}, Accuracy: {accuracy:.2f}%')

    except Exception as e:
        logger.critical(f"An error occurred during training: {str(e)}", exc_info=True)

def test_teacher(teacher, cifar_test, device):
    teacher.eval()
    correct, total = 0, 0
    try:
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(cifar_test):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = teacher(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accuracy = 100. * correct / total

        logger.info(f'Accuracy: {accuracy:.2f}%')

        return accuracy

    except Exception as e:
        logger.critical(f"An error occurred during evaluation: {str(e)}", exc_info=True)
        return 0.0