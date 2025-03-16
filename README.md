# Knowledge Distillation for CIFAR

This repository contains an implementation of knowledge distillation using PyTorch, applied to the CIFAR dataset. Knowledge distillation is a technique for model compression where a smaller "student" model is trained to mimic a larger "teacher" model, enabling better performance than would be possible by training the student model directly on the data.

## Project Structure

```
├── data/                  # CIFAR dataset storage
├── models/                # Saved model files
├── src/                   # Source code
│   ├── data_loader.py     # Data loading functions
│   ├── distillation_loss.py # KD loss implementation
│   ├── models.py          # Teacher and student models
│   ├── train_knowledge_distillation.py
│   ├── train_student.py
│   └── train_teacher.py
├── logs.log               # Training logs
├── main.py                # Main script
├── requirements.txt       # Requirements
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch 2.31+
- torchvision

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/emintiras/knowledge-distillation.git
   cd knowledge-distillation
   ```

2. Install the required packages:
   ```
   pip install torch torchvision
   ```

## Usage

Run the main script to train both the teacher and student models:

 ```
 python main.py
 ```

By default, this will:
1. Train a teacher model (or load a pre-trained one if available)
2. Train a student model without knowledge distillation
3. Train a student model with knowledge distillation
4. Compare the performance of both student models

## Results

The training results are logged to the console and `logs.log`. Key metrics include:

|   |Accuracy|   | 
|---|---|---|
|Teacher model | 79.44% | 
|Student without KD|  66.52% |
|Student with KD   |  70.05% |  

## References

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton et al., 2015)
