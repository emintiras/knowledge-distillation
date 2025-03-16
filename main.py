import torch
import logging
import os
from src.data_loader import get_data_loaders
from src.models import TeacherModel, StudentModel
from src.train_teacher import train_teacher, test_teacher
from src.train_knowledge_distillation import train_knowledge_distillation, test_knowledge_distillation
from src.train_student import train_student, test_student
from src.distillation_loss import distillation_loss

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler("logs.log")
    ]
)

logger = logging.getLogger(__name__)

def main(load_pretrained=False, train_teacher_model=False):
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Get data via loaders
        loaders = get_data_loaders(batch_size=128)
        cifar_train = loaders['cifar_train']
        cifar_test = loaders['cifar_test']
        logger.info("Data loaders initialized.")

        # Create directory for saving models if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Step 1: Get the teacher model (either load pretrained or train from scratch)
        if load_pretrained and os.path.exists('models/teacher.pt'):
            try:
                logger.info("Loading pretrained teacher model...")
                teacher = torch.jit.load('models/teacher.pt')
                logger.info("Pretrained teacher model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load pretrained teacher model: {str(e)}", exc_info=True)
                
                if train_teacher_model:
                    logger.info("Training teacher model from scratch...")
                    teacher = TeacherModel().to(device)
                    teacher = train_teacher(teacher=teacher, cifar_train=cifar_train, 
                                        cifar_test=cifar_test, device=device, 
                                        learning_rate=1e-3, num_epochs=10)
                    # Save the teacher model
                    teacher_scripted = torch.jit.script(teacher)
                    teacher_scripted.save('models/teacher.pt')
                    logger.info("Teacher model trained and saved.")
                else:
                    raise RuntimeError("No teacher model available and train_teacher_model is False")
        else:
            if train_teacher_model:
                logger.info("Training teacher model from scratch...")
                teacher = TeacherModel().to(device)
                teacher = train_teacher(teacher=teacher, cifar_train=cifar_train, 
                                    cifar_test=cifar_test, device=device, 
                                    learning_rate=1e-3, num_epochs=10)
                # Save the teacher model
                teacher_scripted = torch.jit.script(teacher)
                teacher_scripted.save('models/teacher.pt')
                logger.info("Teacher model trained and saved.")
            else:
                logger.info("Loading pretrained teacher model...")
                teacher = torch.jit.load('models/teacher.pt')
                logger.info("Pretrained teacher model loaded successfully.")
        
        # Step 2: Train student model WITHOUT knowledge distillation
        logger.info("Training student model WITHOUT knowledge distillation...")
        student_without_kd = StudentModel().to(device)
        student_without_kd = train_student(
            student=student_without_kd, 
            cifar_train=cifar_train, 
            cifar_test=cifar_test,
            device=device, 
            learning_rate=3e-3, 
            num_epochs=10
        )
        
        # Save the student model without KD
        student_without_kd_scripted = torch.jit.script(student_without_kd)
        student_without_kd_scripted.save('models/student_without_kd.pt')
        logger.info("Student model WITHOUT knowledge distillation trained and saved.")
        
        # Step 3: Train student model WITH knowledge distillation
        logger.info("Training student model WITH knowledge distillation...")
        student_with_kd = StudentModel().to(device)
        student_with_kd = train_knowledge_distillation(
            student=student_with_kd,
            teacher=teacher,
            cifar_train=cifar_train,
            learning_rate=3e-3,
            num_epochs=10,
            distillation_loss=distillation_loss,
            device=device
        )
        
        # Save the student model with KD
        student_with_kd_scripted = torch.jit.script(student_with_kd)
        student_with_kd_scripted.save('models/student_with_kd.pt')
        logger.info("Student model WITH knowledge distillation trained and saved.")
        
        # Step 4: Evaluate and compare both models
        logger.info("=== EVALUATION RESULTS ===")
        
        logger.info("Testing student model WITHOUT knowledge distillation:")
        student_without_kd_acc = test_student(student=student_without_kd, cifar_test=cifar_test, device=device)
        
        logger.info("Testing student model WITH knowledge distillation:")
        student_with_kd_acc = test_student(student=student_with_kd, cifar_test=cifar_test, device=device)
        
        logger.info("Testing teacher model (for reference):")
        teacher_acc = test_teacher(teacher=teacher, cifar_test=cifar_test, device=device)
        
        # Print comparison
        logger.info("\n=== COMPARATIVE RESULTS ===")
        logger.info(f"Teacher model accuracy: {teacher_acc:.2f}%")
        logger.info(f"Student WITHOUT KD accuracy: {student_without_kd_acc:.2f}%")
        logger.info(f"Student WITH KD accuracy: {student_with_kd_acc:.2f}%")
        
        # Show improvement due to KD
        improvement = student_with_kd_acc - student_without_kd_acc
        logger.info(f"Improvement due to knowledge distillation: {improvement:.2f}%")
        logger.info(f"Gap to teacher (WITH KD): {teacher_acc - student_with_kd_acc:.2f}%")
        logger.info(f"Gap to teacher (WITHOUT KD): {teacher_acc - student_without_kd_acc:.2f}%")
        
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Set to True if you have a pretrained teacher model
    load_pretrained = True
    
    # Set to True if you want to train the teacher model from scratch
    # (This will be used only if load_pretrained is False or if loading fails)
    train_teacher_model = False
    
    main(load_pretrained=load_pretrained, train_teacher_model=train_teacher_model)