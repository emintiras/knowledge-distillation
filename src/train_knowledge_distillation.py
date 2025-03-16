import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)

def train_knowledge_distillation(student, teacher, cifar_train, device, learning_rate, num_epochs, distillation_loss):
    student = student.to(device)
    teacher = teacher.to(device)
    optimizer_student = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()
    student.train()

    try:
        for epoch in range(num_epochs):
            running_loss = 0.0
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}] started.")
            for i, (inputs, labels) in enumerate(cifar_train):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer_student.zero_grad()

                with torch.no_grad():
                    teacher_logits = teacher(inputs)
                student_logits = student(inputs)

                loss = distillation_loss(student_logits, teacher_logits, labels)
                loss.backward()
                optimizer_student.step()
                running_loss += loss.item()

            logger.info(f'Epoch {epoch+1}, Loss: {running_loss/(i+1):.3f}')
        return student
    except Exception as e:
        logger.critical(f"An error occurred during training: {str(e)}", exc_info=True)

def test_knowledge_distillation(student, cifar_test, device):
    student.eval()
    correct = 0
    total = 0
    try:
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(cifar_test):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = student(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accuracy = 100. * correct / total

        logger.info(f'Accuracy: {accuracy:.2f}%')

        return accuracy

    except Exception as e:
        logger.critical(f"An error occurred during evaluation: {str(e)}", exc_info=True)
        return 0.0