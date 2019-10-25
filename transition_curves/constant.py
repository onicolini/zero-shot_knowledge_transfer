num_classes = 10
adv_steps = 100
num_images = 1000
epsilon = 1
dataset = 'SVHN'
teacher_path = "./" #Change path of the student an the teacher
student_path = "./" 
#a list of tuple (name_student,name_teacher, depth_student, width_student, depth_teacher, width_teacher)
model_names = [("t-40-2-student-40-1","teacher-40-2",40,1,40,2), 
               ("t-40-2-student-16-2","teacher-40-2",16,2,40,2),
               ("t-40-2-student-16-1","teacher-40-2",16,1,40,2),
               ("t-40-1-student-16-2","teacher-40-1",16,2,40,1),
               ("t-40-1-student-16-1","teacher-40-1",16,1,40,1),
               ("t-16-2-student-16-1","teacher-16-2",16,1,16,2)]



