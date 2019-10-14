import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
import sys
sys.path.append('../')

from wideresnet import Wide_ResNet
from generator import Generator
from cifar10utils import getData, test

'''
generator loss:
@output : logits of the student
@output : logits of the teacher

for the KL div as said here https://discuss.pytorch.org/t/kl-divergence-produces-negative-values/16791/4
and here https://discuss.pytorch.org/t/kullback-leibler-divergence-loss-function-giving-negative-values/763/2
the inputs should be logprobs for the output(student) and probabilities for the targets(teacher)

this was very difficult to understand 

'''
def attention(x):
    """
    Taken from https://github.com/szagoruyko/attention-transfer
    :param x = activations
    """
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def attention_diff(x, y):
    """
    Taken from https://github.com/szagoruyko/attention-transfer
    :param x = activations
    :param y = activations
    """
    return (attention(x) - attention(y)).pow(2).mean()


def divergence(student_logits, teacher_logits):
    divergence = F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1))

    return divergence


def KT_loss_generator(student_logits, teacher_logits):

    divergence_loss = divergence(student_logits, teacher_logits)
    total_loss = - divergence_loss

    return total_loss


def KT_loss_student(student_logits, teacher_logits, student_activations, teacher_activations,beta):

    divergence_loss = divergence(student_logits, teacher_logits)
    if beta > 0:
        at_loss = 0
        for i in range(len(student_activations)):
            at_loss = at_loss + beta * attention_diff(student_activations[i], teacher_activations[i])
    else:
        at_loss = 0

    total_loss = divergence_loss + at_loss

    return total_loss



def main(n_batches,lr_gen,lr_stud,batch_size,test_batch_size,g_input_dim,ng,ns,test_freq,beta, t_depth, t_width,
        s_depth, s_width, teacher_file, student_file, device):
    
    # Print info experiment
    print('Architecture teacher : WRN-' + str(t_depth) + '-' + str(t_width) + ' , from file ' + str(teacher_file))
    print('Architecture student : WRN-' + str(s_depth) + '-' + str(s_width))
    print('Batches: ' + str(n_batches) + ' , batch_size ' + str(batch_size) + ' , test_batch_size ' + str(test_batch_size))
    print('Student lr: ' + str(lr_gen) + ' , Generator lr: ' + str(lr_gen) + ' , Beta: ' + str(beta))
    print('Ng: ' + str(ng) + ' , Ns: ' + str(ns))
    print('Test_freq: ' + str(test_freq) + ' , g_input_dim' + str(g_input_dim))
    print('Device: ' + str(device))
    
    # Get the data
    train_loader, val_loader, test_loader = getData(batch_size,test_batch_size,0.1)
    
    # Get the teacher
    teacher = Wide_ResNet(t_depth,t_width,0,10)
    teacher = teacher.to(device)
    teacher.load_state_dict(torch.load(teacher_file))
    
    # Create the generator
    generator = Generator(z_dim=g_input_dim)
    generator = generator.to(device)
    generator.train()
    
    # Create the student
    student = Wide_ResNet(s_depth,s_width,0,10)
    student = student.to(device)
    
    # Create optimizers (Adam) and LRschedulers(CosineAnnealing) for the generator and the student
    generator_optim = torch.optim.Adam(generator.parameters(), lr=lr_gen)
    gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(generator_optim, n_batches)
    
    student_optim = torch.optim.Adam(student.parameters(), lr=lr_stud)
    stud_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(generator_optim, n_batches)
    
    print('Teacher net test:')
    test_loss, test_accuracy = test(test_loader,teacher,device)
    teacher.eval()
    print('\t Test loss: \t {:.6f}, \t Test accuracy \t {:.2f}'.format(test_loss, test_accuracy))
    
    print('Student net test:')
    test_loss, test_accuracy = test(test_loader,student,device)
    print('\t Test loss: \t {:.6f}, \t Test accuracy \t {:.2f}'.format(test_loss, test_accuracy))
    
    print('Starting training')
    
    for i in range(n_batches):
        print('Batch ' + str(i))
        noise = torch.randn(batch_size,g_input_dim)
        noise = noise.to(device)
        
        gen_loss_print = 0
        
        for j in range(ng):
            gen_imgs = generator(noise)
            gen_imgs = gen_imgs.to(device)

            teacher_pred, *teacher_activations = teacher(gen_imgs)
            student_pred, *student_activations = student(gen_imgs)

            gen_loss = KT_loss_generator(student_pred,teacher_pred)
            generator_optim.zero_grad()
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5)

            generator_optim.step()
            
            gen_loss_print += gen_loss.item()
        
        print('Gen loss :' + str(gen_loss_print/ng) )
        
        stud_loss_print = 0
        for j in range(ns):
            student.train()
            gen_imgs = generator(noise)
            teacher_pred, *teacher_activations = teacher(gen_imgs)
            student_pred, *student_activations = student(gen_imgs)
            
            stud_loss = KT_loss_student(student_pred,teacher_pred, student_activations,teacher_activations, beta )
            student_optim.zero_grad()
            stud_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
            student_optim.step()
            
            stud_loss_print += stud_loss.item()
        
        print('Stud loss :' + str(stud_loss_print/ns) )
            
        stud_scheduler.step()
        gen_scheduler.step()
        
        if(i % test_freq) == 0:
            print('Student net test:')
            test_loss, test_accuracy = test(test_loader,student,device)
            print('\t Test loss: \t {:.6f}, \t Test accuracy \t {:.2f}'.format(test_loss, test_accuracy))
            print('Saving')
            torch.save(student.state_dict(),student_file)
            
    print('Finished and saving')
    torch.save(student.state_dict(),student_file)
            
    
n_batches = 80001
lr_gen = 2e-3
lr_stud = 2e-3
batch_size = 128
test_batch_size = 128
g_input_dim = 100
ng = 1
ns = 10
test_freq = 100
beta = 250
t_depth = 40
t_width = 2
s_depth = 40
s_width = 1
teacher_file = '../pretrained_models/teacher-40-2.pth'
student_file = '../trained_students/student-' + str(s_depth) + '-' + str(s_width) + '.pth'
device = 'cuda:0'
    
main(n_batches,lr_gen,lr_stud,batch_size,test_batch_size,g_input_dim,ng,ns,test_freq,beta, t_depth, t_width,
        s_depth, s_width, teacher_file, student_file, device)
    