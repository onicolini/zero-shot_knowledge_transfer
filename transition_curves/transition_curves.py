from torchvision import models
from torchvision import datasets
from torchvision import transforms
from torchsummary import summary
import torch.nn as nn
from advanceddeep.our_code.Pytorch.wideresnet import Wide_ResNet
from advanceddeep.modifiedcode.models.wresnet import WideResNet
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import csv
import constant as cons 

#pathA = "./advanceddeep/modifiedcode/pretrained/ZeroShot/SVHN/WRN-40-2_to_WRN-16-1/last.pth.tar" 
#pathB = "./advanceddeep/modifiedcode/pretrained/SVHN/WRN-40-2/last.pth.tar"


for nomi in cons.model_names:

    modelA_name = nomi[0]
    modelB_name = nomi[1]

    print("Transition curves di ",modelA_name," e ",modelB_name)
    #MODEL A = STUDENT

    #modelA = Wide_ResNet(nomi[2],nomi[3],0,10)
    modelA = WideResNet(depth=nomi[2], num_classes=10, widen_factor=nomi[3], dropRate=0.0)
    modelA.cuda()
    modelA.load_state_dict(torch.load( cons.student_path+modelA_name+".pth",map_location='cuda:0'))
    modelA.eval()



    #MODEL B = TEACHER

    #modelB = Wide_ResNet(nomi[4],nomi[5],0,10)
    modelB = WideResNet(depth=nomi[4], num_classes=10, widen_factor=nomi[5], dropRate=0.0)
    modelB.cuda()
    modelB.load_state_dict(torch.load( cons.teacher_path+modelB_name+".pth",map_location='cuda:0'))
    modelB.eval()



    # print("modelA")
    # summary(modelA, input_size=(3, 32, 32))
    # print("modelB")
    # summary(modelB, input_size=(3, 32, 32))
    # print(modelA)
    # print(modelB)



    root = "./data/"

    if cons.dataset == 'CIFAR':
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        testset = datasets.CIFAR10(root=root, 
                                    train=False,
                                    download=True, 
                                    transform=transform)
        testloader = torch.utils.data.DataLoader(testset, 
                                    batch_size=1,
                                    shuffle=False)
    elif cons.dataset == 'SVHN':
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])
        testset = datasets.SVHN(root=root, 
                                    split='test',
                                    download=True, 
                                    transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                    batch_size=1,
                                    drop_last=False,
                                    shuffle=False)
    elif cons.dataset == 'MNIST':
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])
        testset = datasets.MNIST(root=root, 
                                    split='test',
                                    download=True, 
                                    transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                    batch_size=1,
                                    drop_last=False,
                                    shuffle=False)



    #f= open("./transition_curves_"+modelA_name+"_"+modelB_name+".csv",'w')

    #fieldnames = ['model','class_i','class_j','p_j','k'] #modello, classe iniziale, classe finale, probabilita classe finale, k" 
    #writer = csv.DictWriter(f,fieldnames=fieldnames)
    #writer.writeheader()
    #model.load_state_dict(torch.load(PATH))
    #mode.eval
    sumsA = [0] * cons.adv_steps
    sumsB = [0] * cons.adv_steps
    c = 0
    MTE = 0

    criterion = nn.CrossEntropyLoss()
    for batch_i, (x,y) in enumerate(testloader) :
        if batch_i >= cons.num_images :
            break
        x = x.to('cuda:0',dtype=torch.float)
        y = y.to('cuda:0')
        print("BATCH I: ", batch_i)
        value , i_a= torch.max(modelA(x)[0].data[0],0)
        #print(i_a,value)
        value , i_b= torch.max(modelB(x)[0].data[0],0)
        #print(i_b, value)
        if i_a == i_b and i_a == y :
            x_0 = x
            MTE_partial_1 = 0
            for j in range(cons.num_classes):
                if( j!=y ):
                    x_adv = x.detach().clone()
                    x_adv.requires_grad = True
                    print(batch_i,j)
                    MTE_partial_0 = 0
                    for k in range(cons.adv_steps):
                        y_a = modelA(x_adv)[0]
                        with torch.no_grad():
                            y_b = modelB(x_adv)[0]
                        
                        loss = criterion(y_a,torch.tensor([j]).to('cuda:0'))
                        #loss = criterion(y_a, y_adversarial)
                        modelA.zero_grad()
                        loss.backward()
                        
                        x_adv.data -= cons.epsilon* x_adv.grad.data
                        x_adv.grad.data.zero_()
                        with torch.no_grad():

                            #print("CURVES A: "+ str( torch.softmax(y_a,dim=1)[0,j] ))
                            #print("CURVES B: "+ str( torch.softmax(y_b,dim=1)[0,j] ),'\n')
                            
                            # writer.writerow({'model': modelA_name, 
                            #              'class_i': y.cpu().numpy()[0], 
                            #              'class_j' : j , 
                            #              'p_j': torch.softmax(y_a,dim=1).data[0][j].cpu().numpy() , 
                            #              'k':k  
                            #             })

                            # writer.writerow({'model': modelB_name, 
                            #              'class_i': y.cpu().numpy()[0], 
                            #              'class_j' : j , 
                            #              'p_j': torch.softmax(y_b,dim=1).data[0][j].cpu().numpy() , 
                            #              'k':k  
                            #             })
                            p_j_A = torch.softmax(y_a,dim=1).data[0][j].cpu().numpy()
                            p_j_B = torch.softmax(y_b,dim=1).data[0][j].cpu().numpy()
                            MTE_partial_0 += abs(p_j_A-p_j_B)
                            sumsA[k] += p_j_A
                            sumsB[k] += p_j_B
                    c += 1
                    MTE_partial_1 += MTE_partial_0 / cons.adv_steps
            MTE = MTE_partial_1 / cons.num_classes
    MTE = MTE / cons.num_images

    #Apro il file delle medie    
    f_media= open("./transition_curves_"+modelA_name+"_student_"+modelB_name+"_teacher.txt",'w')

    #Scrivo le somme del Modello A
    for d in range(cons.adv_steps):
        f_media.write(str(sumsA[d])+",")
    f_media.write("\n")

    #Scrivo le somme del modello B
    for d in range(cons.adv_steps):
        f_media.write(str(sumsB[d])+",")
    f_media.write("\n")

    #Scrivo il contatore di numero samples usati fino ad adesso
    # c = num_samples*(num_classes-1) 
    print("C =",c)
    f_media.write(str(c)+'\n')
    print("MTE =",MTE )
    f_media.write("MTE : "+str(MTE)+'\n')
    f_media.close()
