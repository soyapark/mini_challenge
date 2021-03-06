import gc
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
torch.backends.cudnn.benchmark=True

import numpy as np
import time

import dataset
from models.AlexNet import *
from models.ResNet import *

def run(OPTIMIZATION_OPTION = 0):
    # Parameters
    num_epochs = 20 # do not have to train 10 every time
    output_period = 100
    batch_size = 100

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_34()
    model = model.to(device)

    train_loader, val_loader = dataset.get_data_loaders(batch_size)
    num_train_batches = len(train_loader)

    criterion = nn.CrossEntropyLoss().to(device)
    # TODO: optimizer is currently unoptimized
    # there's a lot of room for improvement/different optimizers
    lr = 1e-3
    lr_step = 1
    optimizer = ''
    scheduler = ''
    if OPTIMIZATION_OPTION == 0: #default
        optimizer = optim.SGD(model.parameters(), lr=1e-3) # change lr value in order to change learning rate 
    elif OPTIMIZATION_OPTION == 3:
        # add your optimization here..
        # optimizer = ..
        lr = 1e-1
        optimizer = optim.SGD(model.parameters(), lr=lr, dampening=0.5, weight_decay=1e-4)
    elif OPTIMIZATION_OPTION == 1:
        lr = 1e-1
        optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=lr, momentum=0.9, dampening=0.5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif OPTIMIZATION_OPTION == 9:
	# starting from low lr and increasing lr at each batch in order to find optimal lr 
        min_lr = 1e-7
        max_lr = 100

        lr = min_lr
        optimizer = optim.SGD(model.parameters(), lr=min_lr,dampening=0.5, weight_decay=1e-4, momentum=0.9)
	
        lr_step=(max_lr/min_lr)**(1/100)
        output_period = 1
    elif OPTIMIZATION_OPTION == 8:
        num_epochs = 9
        # add momentum to option 2
        lr = 1e-1
        optimizer = optim.SGD(model.parameters(), lr=lr, dampening=0.5, weight_decay=1e-4, momentum=0.9)
    elif OPTIMIZATION_OPTION == 5:
        lr = 1e-1
        optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif OPTIMIZATION_OPTION == 6:
        lr = 1e-1
        optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=lr, momentum=0.9, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    else:
     	output_period = 100
    top1_dic = {}
    top5_dic = {}

    #localtime   = time.localtime()
    #timeString  = time.strftime("%Y%m%d%H%M%S", localtime)


    epoch = 1
    while epoch <= num_epochs:
        running_loss = 0.0
        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
        if scheduler:
            scheduler.step()
        model.train()

        for batch_num, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(batch_num)
            #print(inputs)
            #print(labels)                       
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            topk = 5
            #val, index = outputs[0].topk(topk, 0, True, True)
            #print(val, index) 
            
            optimizer.step()
            running_loss += loss.item()

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f' % (
                    epoch, batch_num*1.0/num_train_batches,
                    running_loss/output_period
                    ))
                running_loss = 0.0
                gc.collect()

            # update learning rate every batch in order to get the best lr
	    # ref: Cyclical Learning Rates for Training Neural Networks 	   
            if lr_step != 1:
                print(lr)
            #print()
            lr = lr * lr_step
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        gc.collect()
        # save after every epoch
        torch.save(model.state_dict(), "models/model.%d" % epoch)

        # TODO: Calculate classification error and Top-5 Error
        # on training and validation datasets here
        # comparing the labels vector against the output vector? 
        print("Validating validation datasets ... ")
        model.eval()
        top1_list_val = []
        top5_list_val = []
        for batch_num, (inputs, labels) in enumerate(val_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1_list_val.append(acc1[0] / inputs.size(0))
            top5_list_val.append(acc5[0] / inputs.size(0))

        top1_dic_val[epoch] = np.mean(top1_list_val)
        top5_dic_val[epoch] = np.mean(top5_list_val)

        print("The current model" + str(epoch) + ":")
        print("Top 1 Accuracy: " + str(np.mean(top1_list_val)))
        print("Top 5 Accuracy: " + str(np.mean(top5_list_val)))

        gc.collect()

        if np.mean(top5_list_val) > 0.70:
            print("Predicting test datasets cuz it above 70.0% ... ")
            model.eval()
            _, test_loader = dataset.get_val_test_loaders(batch_size)

            localtime   = time.localtime()
            timeString  = time.strftime("%m%d%H%M%S", localtime)

            f = open("output%s.txt" % timeString, "w")

            for batch_num, (inputs, labels) in enumerate(test_loader, 1):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                for i in range(batch_size):    
                    _, index = outputs[i].topk(5, 0, True, True)
                    f.write("test/%08d.jpg %d %d %d %d %d \n" % ((batch_num-1)*batch_size+(i+1), 
                        index[0], index[1], index[2], index[3], index[4]))

            f.close()
        
        gc.collect()
        epoch += 1


    val_loader2, test_loader= dataset.get_val_test_loaders(batch_size)

    localtime   = time.localtime()
    timeString  = time.strftime("%m%d%H%M%S", localtime)
    f = open("output%s.txt" % timeString, "w")
    #f.write("Woops! I have deleted the content!")

    # test data set
    for batch_num, (inputs, labels) in enumerate(test_loader, 1):
        inputs = inputs.to(device)
        labels = labels.to(device)    
	#print(labels)
        outputs = model(inputs)
        
        for i in range(batch_size):    
            val, index = outputs[i].topk(5, 0, True, True)
            f.write("test/%08d.jpg %d %d %d %d %d \n" % ((batch_num-1)*batch_size+(i+1), index[0], index[1], index[2], index[3], index[4]))

    f.close()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
 
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
 
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

print('Starting training')
OPTIMIZATION_COUNT = 10
for i in range(8,OPTIMIZATION_COUNT):
	print("************ Running optimization %d" % i)
	run(i)
print('Training terminated')
print()
