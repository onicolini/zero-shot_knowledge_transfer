# Advanced Deep Learning
Advanced Deep Learning project to do during Period 1 (September and October) 2019 an KTH.
The task is to reproduce and reimplement the paper "Zero-shot Knowledge Transfer via Adversarial Belief Matching" (https://arxiv.org/pdf/1905.09768.pdf) presented at NIPS 2019. 
The project has been done by Olivier Nicolini, Matteo Tadiello and Simone Zamboni.

# Author's code
The first folder is the "Authors' code". There are two more subfolders. 
In the first there is the Few shot code (taken from https://github.com/polo5/FewShotKnowledgeTransfer) for the Knowledge Distillation + Attention used in the paper as a baseline against the zero-shot method. In this folder we only downloaded the pretrained models and created a file called our main, that we used with command line options in order to run the six experiments with CIFAR10.
In the second folder there is the code for the zero-shot learning (taken from https://github.com/polo5/ZeroShotKnowledgeTransfer), where we downloaded the pretrained models, we fixed a call to function in the logger given by an update in the library and we created seven files, corresponding to the six experiments on CIFAR10 and one on SVHN.

# Our code
Our code is split in two.
First we can find the Keras folder, that contains our trials to implement the zero-shot algorithm on Keras. We found very difficult to get the intermediate activations and train the generation with backpropagation trought both the student and the teacher, so we decided to switch framework to Pytorch. This was still a valuable learning experience and we left our not working code in there if someone wants to do what we failed on doing.
Then there is the Pytorch folder. In the main folder we can find 4 files:
 - cifar10utils.py: utils to download, get the dataloader and test a network on CIFAR10
 - generator.py: the generator network, taken directly from the authors' code
 - wideresnet.py: a modification of https://github.com/indussky8/wide-resnet.pytorch/blob/master/networks/wide_resnet.py to have a WideResNetwork with also the activations as output
 - zero-shot-baseline.py: the complete code to run a zero-shot algorithm from a pretrained models. All the experiments use a variation of this file
These four files are all you need to run a zero-shot algorithm. Modifying zero-shot-baseline.py we created our experiments and we divide the in folders 
Then we find 5 folders:
 - pretrained_models: contains the teachers with the code used to create them
 - Basic_experiments: contrains the code to run the six experiments on the CIFAR10 of the paper, with in the folder trained_students the resulting students, to use them for the transition curves
- Advanced_experiments: contrains the code to run the advanced experiments (different betas ecc..) on the CIFAR10, with in the folder trained_students the resulting students, to use them for the transition curves
 - Other_dataset_experiments: contains the code to another WideResNet that accepts as input one channel images for MINST(and FashionMNIST), and also a generator that creates one channel images for MNIST (and FashionMNIST), then the code to train teachers on MNIST, FashionMNIST, SVHN and CIFAR100, the trained teachers and the code to run a zero-shot on each of these teachers. All the resulting students (and generators) are saved in the folder trained_students, to use them for the transition curves
Due to the fact that we have saved all the pretrained models, both ours and from the authors, the Github folder is quite heavy.
