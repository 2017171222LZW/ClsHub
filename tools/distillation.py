# About Model Distillation
# Dataset --> model_teacher
# teacher = model_teacher + softmax with high temperature
# Dataset + teacher --> student_model
# student = student_model + softmax with normal temperature
# target is student with the similar ability as the teachers
