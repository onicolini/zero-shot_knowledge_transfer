#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ATTENTION READ THIS CAREFOULLY!

# before runnig this modify the file logger.py replacing the scipy.misc.toImage()
# function as described here: 
# https://stackoverflow.com/questions/57545125/attributeerror-module-scipy-misc-has-no-attribute-toimage
# then close, shutdonw this notebook and reopen it

# this file should be in the main folder of the original code from github
# you also have to download the pretrained models and put them in a folder in this folder called pretrained

# moreover you shoud create a folder for the datasets, one for the model and one for the logs,
# possibily all in this folder

'''
This is the main modified in order to have arguments manually added instead
of setting them as option when running the program from command line
'''

import torch
from solver import *
from utils.helpers import *

class Myargs():
    pass

def main():
    """
    Run the experiment as many times as there
    are seeds given, and write the mean and std
    to as an empty file's name for cleaner logging
    """
    
    # insert experiment options
    args = Myargs()
    args.dataset = "CIFAR10"
    args.total_n_pseudo_batches = 80000
    args.n_generator_iter = 1
    args.n_student_iter = 10
    args.batch_size = 128
    args.z_dim = 100
    args.student_learning_rate = 2e-3
    args.generator_learning_rate = 1e-3
    args.teacher_architecture = 'WRN-40-2'
    args.student_architecture = 'WRN-16-1'
    args.KL_temperature = 1
    args.AT_beta = 250
    
    args.pretrained_models_path = "./pretrained/"
    args.datasets_path = "./datasets/"
    args.log_directory_path = "./log/"
    args.save_final_model = 0
    args.save_n_checkpoints = 0
    args.save_model_path = "./save_model/"
    args.seeds = [0] 
    args.workers = 2
    args.use_gpu = True    
    
    args.dataset_path = "./datasets/"
    args.log_freq = 100
    
    args.experiment_name = "test"   
    args.device = torch.device('cuda:0')
   
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if len(args.seeds) > 1:
        test_accs = []
        base_name = args.experiment_name
        for seed in args.seeds:
            print('\n\n----------- SEED {} -----------\n\n'.format(seed))
            set_torch_seeds(seed)
            args.experiment_name = os.path.join(base_name, base_name+'_seed'+str(seed))
            solver = ZeroShotKTSolver(args)
            test_acc = solver.run()
            test_accs.append(test_acc)
        mu = np.mean(test_accs)
        sigma = np.std(test_accs)
        print('\n\nFINAL MEAN TEST ACC: {:02.2f} +/- {:02.2f}'.format(mu, sigma))
        file_name = "mean_final_test_acc_{:02.2f}_pm_{:02.2f}".format(mu, sigma)
        with open(os.path.join(args.log_directory_path, base_name, file_name), 'w+') as f:
            f.write("NA")
    else:
        set_torch_seeds(args.seeds[0])
        solver = ZeroShotKTSolver(args)
        test_acc = solver.run()
        print('\n\nFINAL TEST ACC RATE: {:02.2f}'.format(test_acc))
        file_name = "final_test_acc_{:02.2f}".format(test_acc)
        with open(os.path.join(args.log_directory_path, args.experiment_name, file_name), 'w+') as f:
            f.write("NA")
            
            
"""
OLD CODE FOR RUNNING THE PROGRAM FROM COMMAND LINE
if __name__ == "__main__":
    import argparse
    import numpy as np
    from utils.helpers import str2bool
    print('Running...')

    parser = argparse.ArgumentParser(description='Welcome to the future')

    parser.add_argument('--dataset', type=str, default='SVHN', choices=['SVHN', 'CIFAR10'])
    parser.add_argument('--total_n_pseudo_batches', type=float, default=1000)
    parser.add_argument('--n_generator_iter', type=int, default=1, help='per batch, for few and zero shot')
    parser.add_argument('--n_student_iter', type=int, default=7, help='per batch, for few and zero shot')
    parser.add_argument('--batch_size', type=int, default=128, help='for few and zero shot')
    parser.add_argument('--z_dim', type=int, default=100, help='for few and zero shot')
    parser.add_argument('--student_learning_rate', type=float, default=2e-3)
    parser.add_argument('--generator_learning_rate', type=float, default=1e-3)
    parser.add_argument('--teacher_architecture', type=str, default='LeNet')
    parser.add_argument('--student_architecture', type=str, default='LeNet')
    parser.add_argument('--KL_temperature', type=float, default=1, help='>1 to smooth probabilities in divergence loss, or <1 to sharpen them')
    parser.add_argument('--AT_beta', type=float, default=250, help='beta coefficient for AT loss')

    parser.add_argument('--pretrained_models_path', nargs="?", type=str, default='/home/paul/Pretrained/')
    parser.add_argument('--datasets_path', type=str, default="/home/paul/Datasets/Pytorch/")
    parser.add_argument('--log_directory_path', type=str, default="/home/paul/git/ZeroShotKnowledgeTransfer/logs/")
    parser.add_argument('--save_final_model', type=str2bool, default=0)
    parser.add_argument('--save_n_checkpoints', type=int, default=0)
    parser.add_argument('--save_model_path', type=str, default="/home/paul/git/FewShotKT/logs/")
    parser.add_argument('--seeds', nargs='*', type=int, default=[0, 1])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--use_gpu', type=str2bool, default=False, help='set to False to debug on cpu, using LeNets')
    args = parser.parse_args()

    args.total_n_pseudo_batches = int(args.total_n_pseudo_batches)
    if args.AT_beta > 0: assert args.student_architecture[:3] in args.teacher_architecture
    args.log_freq = max(1, int(args.total_n_pseudo_batches / 100))
    args.dataset_path = os.path.join(args.datasets_path, args.dataset)
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    args.device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    args.experiment_name = 'ZEROSHOTKT_{}_{}_{}_gi{}_si{}_zd{}_plr{}_slr{}_bs{}_T{}_beta{}'.format(args.dataset, args.teacher_architecture,  args.student_architecture, args.n_generator_iter, args.n_student_iter, args.z_dim, args.generator_learning_rate, args.student_learning_rate, args.batch_size, args.KL_temperature, args.AT_beta)

    print('\nTotal data batches: {}'.format(args.total_n_pseudo_batches))
    print('Logging results every {} batch'.format(args.log_freq))
    print('\nRunning on device: {}'.format(args.device))
"""

main()


# In[ ]:




