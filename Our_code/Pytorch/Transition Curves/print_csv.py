# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv 
import re
import os
import constant as cons

# file_path = "./transition_curves_"+modelA_name+"_"+modelB_name+".csv"

files = os.listdir('./results')

for x in files:
    if (re.match("...png",x)):
        os.remove(os.path.join('./results/',x))
        print("Deleted "+x)


# with open(file_path) as csv_file:
#     csv_reader = csv.reader(csv_file,delimiter=',')
#     line_count = 0
#     class_i_j = []
#     sample_model_A = [[]]
#     sample_model_B = [[]]
#     for row in csv_reader:
#         if line_count == 0 :
#             print(row)
#             line_count += 1
#         else:
#             index = 0
#             try:
#                 index = class_i_j.index(""+row[1]+row[2])
#             except:
#                 class_i_j.append(""+row[1]+row[2])
#                 sample_model_A.append([])
#                 sample_model_B.append([])
#                 index = len(class_i_j)-1
#             if (row[0] == modelA_name):
#                 sample_model_A[index].append([float(row[3])])
#             if (row[0] == modelB_name):
#                 sample_model_B[index].append([float(row[3])])
#             line_count += 1

# #print(class_i_j,sample_model_1)
# #print(class_i_j,sample_model_2)
# for i in range(len(class_i_j)):
#     fig, ax = plt.subplots()
#     ax.plot(sample_model_A[i],label=modelA_name)
#     ax.plot(sample_model_B[i],label=modelB_name)
#     ax.legend()
#     plt.savefig("./results/"+class_i_j[i]+".png")

for nomi in cons.model_names:

    modelA_name = nomi[0]
    modelB_name = nomi[1]

    f =open("transition_curves_"+modelA_name+"_student_"+modelB_name+"_teacher.txt",'r')
    fig2, ax2 = plt.subplots()
    somme_A = f.readline().split(',')
    del somme_A[-1]
    print(somme_A)
    somme_B = f.readline().split(',')
    del somme_B[-1]
    c = int(f.readline())
    for a in range(len(somme_A)):
        somme_A[a] = float(somme_A[a])/c
        print(a)
    for b in range(len(somme_B)):
        somme_B[b] = float(somme_B[b])/c
    print(somme_A)
    ax2.plot(somme_A,label=modelA_name)
    ax2.plot(somme_B,label=modelB_name)
    ax2.legend()
    plt.savefig("Avarage_curves_"+modelA_name+"_"+cons.dataset+".png")
    f.close()
