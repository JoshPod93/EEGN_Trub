'''
EEGNet re-build function bank.
'''

import pandas as pd
import numpy as np
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score

import matplotlib.pyplot as plt

'-------------------------------------------------------------------------------------------------'
'--------------------------------General Functions------------------------------------------------'
'-------------------------------------------------------------------------------------------------'


def data_info(X_train, y_train, X_val, y_val, X_test, y_test):
    '''
    Print out info on datasets and labels used in network optimization.

    NOTE: Assumes info from train, val and test datasets.

    Example: data_info(X_train, y_train, X_val, y_val, X_test, y_test)
    '''

    print('------Data Info')
    print('X_train : ', X_train.shape)
    print('y_train : ', y_train.shape)
    print('X_val : ', X_val.shape)
    print('y_val : ', y_val.shape)
    print('X_test : ', X_test.shape)
    print('y_test : ', y_test.shape)
    print('\n')


def rand_data_gen(num_events, num_samps, num_chans, verbose):
    '''
    Generate randomised data for train, val and test datasets.

    NOTE: Creates matrices of dimensions: Events x Samples x Singleton x Channels.

    Example: X_train, X_val, X_test = rand_data_gen(num_events=1000, num_samps=120, num_chans=2, verbose=1)
    '''
    # Generate randomised values between 0 and 1 using: np.random.rand.
    X_train = np.random.rand(num_events, 1, num_samps, num_chans).astype('float32')
    X_val = np.random.rand(num_events, 1, num_samps, num_chans).astype('float32')
    X_test = np.random.rand(num_events, 1, num_samps, num_chans).astype('float32')

    if verbose == 1:
        print('X_train: ', X_train.shape, X_train[0, 0, 0:5, 0])
        print('X_val: ', X_val.shape, X_val[0, 0, 0:5, 0])
        print('X_test: ', X_test.shape, X_test[0, 0, 0:5, 0])

    return X_train, X_val, X_test


def rand_lab_gen(num_events, verbose):
    '''
    Generate randomised lists of 0's and 1's for train, val and test data sets.

    NOTE: try using a factor of 2 for division of value.

    Example: y_train, y_val, y_labels = rand_lab_gen(num_events=1000, verbose=1)
    '''

    # Create list of exactly half 0's and half 1's.
    zeros = np.zeros((np.int(np.round(num_events/2))))
    ones = np.ones((np.int(np.round(num_events/2))))
    # Append 0's and 1's to singluar list.
    labels = np.append(zeros, ones)
    y_train = np.append(zeros, ones)
    y_val = np.append(zeros, ones)
    y_test = np.append(zeros, ones)
    # Convert to Integer format.
    labels = labels.astype(int)
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)
    # Randomize Train Labels.
    random.shuffle(y_train)
    # Randomize Val Labels.
    random.shuffle(y_val)
    # Randomize Test Labels.
    random.shuffle(y_test)

    if verbose == 1:
        print('Labels: ', labels[0:5])
        print('y_train: ', y_train[0:5])
        print('y_val: ', y_val[0:5])
        print('y_test: ', y_test[0:5])

    return y_train, y_val, y_test


def live_plotter(x_vec, y1_data, line1, identifier='', pause_time=0.1):
    'REF: https://github.com/ideoforms/pylive'
    # use ggplot style for more sophisticated visuals
    plt.style.use('ggplot')
    '''REF: https://github.com/makerportal/pylive/blob/master/pylive.p'''
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)
        #update plot label/title
        plt.xlabel('Epochs')
        plt.title('Training Performance Plot: {}'.format(identifier))
        plt.show()
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    # return line so we can update it again in the next iteration
    return line1


def live_plot_dual(x_vec, y1_data, y2_data, line1, line2, identifier='', pause_time=0.1):
    'REF: https://github.com/ideoforms/pylive'
    # use ggplot style for more sophisticated visuals
    plt.style.use('ggplot')
    '''REF: https://github.com/makerportal/pylive/blob/master/pylive.p'''
    if line1==[]:
        '---Loss Line'
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(121)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec, y1_data, '-x', alpha=0.5)
        #update plot label/title
        plt.xlabel('Epochs')
        plt.title('Loss')
        plt.show()

        '---Accuracy Line'
        ax2 = fig.add_subplot(122)
        # create a variable for the line so we can later update it
        line2, = ax2.plot(x_vec, y2_data, '-x', alpha=0.5)
        #update plot label/title
        plt.xlabel('Epochs')
        plt.title('Accuracy')
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    line2.set_ydata(y2_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])

    if np.min(y2_data)<=line2.axes.get_ylim()[0] or np.max(y2_data)>=line2.axes.get_ylim()[1]:
        plt.ylim([np.min(y2_data)-np.std(y2_data),np.max(y2_data)+np.std(y2_data)])

    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    # return line so we can update it again in the next iteration
    return line1, line2


def live_plotter_xy(x_vec, y1_data, line1, identifier='', pause_time=0.01):
    # the function below is for updating both x and y values (great for updating dates on the x-axis)
    # use ggplot style for more sophisticated visuals
    plt.style.use('ggplot')
    if line1==[]:
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        line1, = ax.plot(x_vec,y1_data,'r-o',alpha=0.8)
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()
    line1.set_data(x_vec,y1_data)
    plt.xlim(np.min(x_vec),np.max(x_vec))
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    plt.pause(pause_time)
    return line1


def live_plotter_xy_dual(x_vec, y_loss, y_acc, loss_line, acc_line, pause_time=0.01):
    # the function below is for updating both x and y values (great for updating dates on the x-axis)
    # use ggplot style for more sophisticated visuals
    plt.style.use('ggplot')
    if loss_line==[]:
        'Loss'
        plt.ion()
        fig = plt.figure(figsize=(8, 5))
        fig.suptitle('Training Performance')
        ax = fig.add_subplot(121)
        loss_line, = ax.plot(x_vec, y_loss, 'r-x', alpha=0.6, markersize=4)
        plt.xlabel('Epochs')
        plt.title('Loss')
        'Accuracy'
        ax2 = fig.add_subplot(122)
        acc_line, = ax2.plot(x_vec, y_acc, 'b-x', alpha=0.6, markersize=4)
        plt.title('Accuracy (%)')
        plt.xlabel('Epochs')
        plt.show()

    '---Loss Update'
    plt.subplot(121)
    loss_line.set_data(x_vec, y_loss)
    plt.xlim(np.min(x_vec) - 1, np.max(x_vec) + 1)

    if np.min(y_loss)<=loss_line.axes.get_ylim()[0] or np.max(y_loss)>=loss_line.axes.get_ylim()[1]:
        plt.ylim([np.min(y_loss)-np.std(y_loss),np.max(y_loss)+np.std(y_loss)])

    '---Accuracy Update'
    plt.subplot(122)
    acc_line.set_data(x_vec, y_acc)
    plt.xlim(np.min(x_vec) - 1, np.max(x_vec) + 1)

    if np.min(y_acc)<=acc_line.axes.get_ylim()[0] or np.max(y_acc)>=acc_line.axes.get_ylim()[1]:
        plt.ylim([np.min(y_acc)-np.std(y_acc),np.max(y_acc)+np.std(y_acc)])

    plt.pause(pause_time)

    return loss_line, acc_line


def live_plotter_xy_uber(x_vec, y_loss, y_acc, y_v_acc, loss_line, acc_line, v_acc_line, pause_time=0.01):
    # the function below is for updating both x and y values (great for updating dates on the x-axis)
    # use ggplot style for more sophisticated visuals
    plt.style.use('ggplot')
    if loss_line==[]:
        'Loss'
        plt.ion()
        fig = plt.figure(figsize=(8, 5))
        fig.suptitle('Training Performance')
        ax = fig.add_subplot(121)
        loss_line, = ax.plot(x_vec, y_loss, 'r-x', alpha=0.6, markersize=4)
        plt.xlabel('Epochs')
        plt.title('Loss')
        'Accuracy'
        ax2 = fig.add_subplot(122)
        acc_line, = ax2.plot(x_vec, y_acc, 'b-x', alpha=0.6, markersize=4)
        v_acc_line, = ax2.plot(x_vec, y_v_acc, 'g-x', alpha=0.6, markersize=4)
        plt.title('Accuracy (%)')
        plt.xlabel('Epochs')
        plt.show()

    '---Loss Update'
    f = plt.figure(1)
    a = f.axes
    f.axes[0]
    '---Assign Data'
    loss_line.set_data(x_vec, y_loss)
    '---X Axis'
    f.axes[0].set_xlim(np.min(x_vec) - 1, np.max(x_vec) + 1)
    '---Y Axis'
    f.axes[0].set_ylim(np.min(y_loss) - 0.1, np.max(y_loss) + 0.1)
    # if np.min(y_loss)<=loss_line.axes.get_ylim()[0] or np.max(y_loss)>=loss_line.axes.get_ylim()[1]:
    #     plt.ylim([np.min(y_loss)-np.std(y_loss),np.max(y_loss)+np.std(y_loss)])
    '---Plot Pause'
    plt.pause(pause_time)

    '---Training Accuracy Update'
    f.axes[1]
    '---Assign Data'
    acc_line.set_data(x_vec, y_acc)
    '---X Axis'
    f.axes[1].set_xlim(np.min(x_vec) - 1, np.max(x_vec) + 1)
    '---Y Axis'
    if np.min(y_acc)<=acc_line.axes.get_ylim()[0] or np.max(y_acc)>=acc_line.axes.get_ylim()[1]:
        plt.ylim([np.min(y_acc)-np.std(y_acc),np.max(y_acc)+np.std(y_acc)])
    '---Testing Accuracy Update'
    '---Assign Data'
    v_acc_line.set_data(x_vec, y_v_acc)
    '---X Axis'
    plt.xlim(np.min(x_vec) - 1, np.max(x_vec) + 1)
    # f.axes[1].set_xlim(np.min(x_vec) - 1, np.max(x_vec) + 1)
    '---Y Axis Combo Update'
    comb_data = np.append(y_acc, y_v_acc)
    if np.min(comb_data)<=v_acc_line.axes.get_ylim()[0] or np.max(comb_data)>=v_acc_line.axes.get_ylim()[1]:
        plt.ylim([np.min(comb_data)-np.std(comb_data),np.max(comb_data)+np.std(comb_data)])
    '---Plot Pause'
    plt.pause(pause_time)

    return loss_line, acc_line, v_acc_line


def gpu_check():
    '''
    Testing if gpu is being utilized by the PyTorch training scheme.
    REF: https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
    '''
    import torch
    torch.cuda.current_device()
    torch.cuda.device(0)
    torch.cuda.device_count()
    print(torch.cuda.get_device_name(0))
    torch.cuda.is_available()


def get_device():
    '''
    Check for cuda compatibile hardware to run PyTorch in GPU ready tensors.
    '''
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print('Device: ', device, '\n')

    return device


def lab_distrib(labels):
    '''
    Prints info on the proportion and amount of classes in a given dataset.

    Example: lab_distrib(labels)

    '''
    print('------Labels Distribution')
    print('Labels DIMS: ', labels.shape, '\n', labels[0:5])
    classes, num = np.unique(labels, return_counts=True)
    print('Classes: ', classes, '| Num Instances: ', num)
    total = sum(num)

    for i in range(len(classes)):
        cP = (num[i] / total) * 100
        print('Class: {0} | Num: {1} | %: {2}'.format(classes[i], num[i], cP))


def net_info(net, optimizer):
    '---Net Info----'
    # REF: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])


def class_balance(X, y, balance, perm, verbose):
    '''

    Generates a class balance of P3 and Non-P3 events based on Binary class labels.

    1) grabs all indices of the P3 and NP3 events from the string labels array and stores separately.
    2) shuffles the NP3 label indices array to ensure sampling across entire session.
    3) cuts down NP3 label indices array to same size as that of the P3 labels indices array for
        a clean 50/50 split to maintain class blance in analysis model fits and testing.
    4) aggregates P3 and non-P3 data according to these indices into separate data arrays.
    5) brings all P3 and non-P3 data and labels together for Striatified Shuffling or randomized
        subsampling further down the analysis pipeline.

    NOTE: Assumes Events x Samples.

    NOTE: All event indicies for P3 and Non-P3 labels are random shuffled to ensure even distibution of
          subject data across training, validtation and test sets.

    Inputs:

    data = aggregated segmented EEG data across session.
    labels = MUST BE BINARY ground truth string labels indicating eeg event P3 / NP3.
    balance = refers to a percentage difference in terms of P3 vs NP3 events in final aggregate array.
                For example, a value of 1 means 1:1 ratio, a value of 2 means a 1:2 / P3:NP3 ratio.
                This has a hard-limit as there are only so many P3s to NP3, recommend max of 5.
    verbose = info on process 1 == print one line essential info on P3 vs NP3 ratio, 2 == print all info.

    Outputs:

    bal_data = aggregated class-balanced data matrix.
    bal_labels = aggregated class-balanced string labels array.
    bal_i_labels = aggregated class-balanced numeric labels array.

    Example:

    bal_data, bal_labels = pb.class_balance(X, y)

    '''
    import random

    # Preform P3 and NP3 agregate arrays.
    p3 = []
    np3 = []
    # Gather all indices of P3 (1) and NP3 (0) events in the labels array.
    id_np3 = [i for i, p in enumerate(y) if p == 0]
    id_p3 = [i for i, p in enumerate(y) if p == 1]
    # Convert to subscriptable numpy array.
    id_np3 = np.asarray(id_np3)
    id_p3 = np.asarray(id_p3)
    # Convert to numeric type integer for indexing.
    id_np3 = id_np3.astype(int)
    id_p3 = id_p3.astype(int)
    # Shuffle the All event indices lists.
    random.shuffle(id_p3)
    random.shuffle(id_np3)
    # Balance Value.
    bal_val = np.int(len(id_p3) * balance)
    # Reduce the number of NP3 indices to the amount examples secified vai the balance value.
    id_np3 = id_np3[0:bal_val]
    # Print function info.
    if verbose == 1 or verbose == 2:
        print('------Class Balance Stats:')
    if verbose == 2:
        print('ID_P3 / NUMBER OF P3 EVENTS: ', len(id_p3))
        print('ID_NP3 / NUMBER OF NP3 EVENTS PRE SHUFFLE: ', len(id_np3))
        print(id_p3[0], len(id_p3), id_p3[0:10])
        print(id_np3[0], len(id_np3), id_np3[0:10])
        print('ID_NP3 / NUMBER OF NP3 EVENTS POST SHUFFLE: ', len(id_np3))
    # Aggregate P3 signals together.
    p3 = X[id_p3, :]
    # Aggregate NP3 signals together to a ratio relative to the P3 class events.
    np3 = X[id_np3, :]
    # Aggregate P3 and NP3 events into single data matrix.
    bal_data = np.append(p3, np3, axis=0)
    # Index into P3 / NP3 label locations and append.
    bal_labels = np.append(y[id_p3], y[id_np3])

    'Final Randomisation.'
    rand_ind = np.arange(np.int(len(id_p3) + len(id_np3)))
    random.shuffle(rand_ind)

    bal_data = bal_data[rand_ind, :, :]
    bal_labels = bal_labels[rand_ind]

    print('RAND INT: ', rand_ind)

    if verbose == 2:
        print('Bal Data DIMS: ', bal_data.shape)
        print('Bal Labels DIMS: ', bal_labels.shape)
    if verbose == 1 or verbose == 2:
        print('P3:NP3 ratio: {0} : {1} / {2} : {3}'.format(1, balance, len(id_p3), len(id_np3)))
        print('\n')

    return bal_data, bal_labels


def tensor_print(x, indicator, printer):
    '''
    Print info on specified (indicator) tensor in neural network.

    Example: tensor_print(x, indicator='BN 1 Output Dims')
    '''
    if printer == True:
        print(indicator + ' : ', x.shape)


def reducer(data, type, ref, chan_list, verbose):
    '''
    Signal Reduction to Samples x Emoji / Event matrix.
    Reduce array down to just one signal either by isolating Cz or averaging across all OR a number of channels.

    Assumes Emoji / Events x Channels x Samples

    Inputs:

    data        = eeg time-series.
    type        = either 'AVG' meaning an average signal is generated based off of the channel indices in chan_list,
                  or 'IND', meaning a single channel e.g. Cz  is simply taken from the list and added to a separate
                  output matrix.
    ref         = matrix channel index for electrode you want to individually ('IND') isolate, if None, then skip.
    chan_list   = list of channel indices for cross-channel averaging.
    verbose     = print function info.

    Outputs:

    out_data    = single channels of emoji level data (Samples x Channels x Emoji Events).
                  Channel dimension is just a singleton dimensions to retain features of following steps.

    IND Example: i_data = reducer(r_data, type='IND', ref=[1], chan_list=None, verbose=1)
    AVG Example: i_data = reducer(r_data, type='AVG', ref=None, chan_list=[0, 1], verbose=1)


    '''

    if verbose == 1:
        print('Pre-Reduction DATA DIMS: ', data.shape)
    if type == 'IND':
        data = data[:, ref, :]
    elif type == 'AVG':
        out_data = np.average(data[:, chan_list, :], axis=1)
        out_data = np.expand_dims(out_data, axis=1)
    if verbose == 1:
        print('Post-Reduction DATA DIMS: ', out_data.shape)

    return out_data


def data_part(eeg, labels, tr_qt, perm, verbose):
    '''
    Function for segregating dataset into training, validation and test datasets.

    Combined with class blance script to ensure NP3 samples are acquired across
    entire dataset, in same way P3 events are subsampled.

    NOTE: verbose = 1 prints one line essential info on partition quantities, 2 = all info printed.

    NOTE: perm == 1 reshapes data arrays for EEGNet style data dim org.

    Example: eeg_train, labels_train, eeg_val, labels_val, eeg_test, labels_test = data_part(eeg, labels, tr_qt=0.75)

    '''

    print('------DATA PART')

    num_classes = np.int(len(np.unique(labels)))

    id_np3 = [i for i, x in enumerate(labels) if x == 0]
    id_p3 = [i for i, x in enumerate(labels) if x == 1]

    n_samps, n_chans, n_events = np.shape(np.squeeze(eeg))

    if verbose == 2:
        print('ID_NP3: ', len(id_np3), id_np3[0:5])
        print('ID_P3: ', len(id_p3), id_p3[0:5])

    '---NP3 Bounds'
    bnd_np3_1 = np.round(np.int(len(id_np3) * tr_qt))
    bnd_np3_2 = bnd_np3_1 + np.round(np.int((len(id_np3) * (1 - tr_qt) / 2)))

    if verbose == 2:
        print('bnd_np3_1: ', bnd_np3_1)
        print('bnd_np3_2: ', bnd_np3_2)

    # NP3 Data
    eeg_train_np3 = eeg[id_np3[0:bnd_np3_1], :, :]
    eeg_val_np3 = eeg[id_np3[bnd_np3_1:bnd_np3_2], :, :]
    eeg_test_np3 = eeg[id_np3[bnd_np3_2:], :, :]

    if verbose == 2:
        print('NP3 EEG DIMS: ', eeg.shape)
        print('NP3 EEG TRAIN: ', eeg_train_np3.shape)
        print('NP3 EEG VAL: ', eeg_val_np3.shape)
        print('NP3 EEG TEST: ', eeg_test_np3.shape)

    # NP3 Labels
    lab_train_np3 = labels[id_np3[0:bnd_np3_1]]
    lab_val_np3 = labels[id_np3[bnd_np3_1:bnd_np3_2]]
    lab_test_np3 = labels[id_np3[bnd_np3_2:]]

    if verbose == 2:
        print('NP3 LABELS TRAIN: ', lab_train_np3.shape)
        print('NP3 LABELS VAL: ', lab_val_np3.shape)
        print('NP3 LABELS TEST: ', lab_test_np3.shape)

    '---P3 Bounds'
    bnd_p3_1 = np.round(np.int(len(id_p3) * tr_qt))
    bnd_p3_2 = bnd_p3_1 + np.round(np.int((len(id_p3) * (1 - tr_qt) / 2)))

    if verbose == 2:
        print('bnd_p3_1: ', bnd_p3_1)
        print('bnd_p3_2: ', bnd_p3_2)

    # P3 Data
    eeg_train_p3 = eeg[id_p3[0:bnd_p3_1], :, :]
    eeg_val_p3 = eeg[id_p3[bnd_p3_1:bnd_p3_2], :, :]
    eeg_test_p3 = eeg[id_p3[bnd_p3_2:], :, :]

    if verbose == 2:
        print('P3 EEG TRAIN: ', eeg_train_p3.shape)
        print('P3 EEG VAL: ', eeg_val_p3.shape)
        print('P3 EEG TEST: ', eeg_test_p3.shape)

    # P3 Labels
    lab_train_p3 = labels[id_p3[0:bnd_p3_1]]
    lab_val_p3 = labels[id_p3[bnd_p3_1:bnd_p3_2]]
    lab_test_p3 = labels[id_p3[bnd_p3_2:]]

    if verbose == 2:
        print('P3 LABELS TRAIN: ', lab_train_p3.shape)
        print('P3 LABELS VAL: ', lab_val_p3.shape)
        print('P3 LABELS TEST: ', lab_test_p3.shape)

    '---Compound'
    # Data
    eeg_train = np.append(eeg_train_np3, eeg_train_p3, axis=0)
    eeg_val = np.append(eeg_val_np3, eeg_val_p3, axis=0)
    eeg_test = np.append(eeg_test_np3, eeg_test_p3, axis=0)

    if verbose == 1:
        print('EEG TRAIN: ', eeg_train.shape)
        print('EEG VAL: ', eeg_val.shape)
        print('EEG TEST: ', eeg_test.shape)

    # Labels
    labels_train = np.append(lab_train_np3, lab_train_p3)
    labels_val = np.append(lab_val_np3, lab_val_p3)
    labels_test = np.append(lab_test_np3, lab_test_p3)

    if verbose == 2:
        print('LABELS TRAIN: ', labels_train.shape)
        print('LABELS VAL: ', labels_val.shape)
        print('LABELS TEST: ', labels_test.shape)

    if perm == 1:
        eeg_train = np.moveaxis(eeg_train, 2, 3).astype('float32')
        eeg_val = np.moveaxis(eeg_val, 2, 3).astype('float32')
        eeg_test = np.moveaxis(eeg_test, 2, 3).astype('float32')

    'Convert to Float 32'
    eeg_train = eeg_train.astype('float32')
    eeg_val = eeg_val.astype('float32')
    eeg_test = eeg_test.astype('float32')

    print('\n')
    return eeg_train, labels_train, eeg_val, labels_val, eeg_test, labels_test


def w8_lrn_check(pre_train_w8, post_train_w8, loop, verbose):
    '''
    Simply uses torch.eq to test if the model is training, compares torch tensor
    elements for equivalence within the first 10 elements.

    NOTE: must use weights from the same layer in the network.

    NOTE: loop denotes whether the check is being performed in the Training (1) or Test / Val stages (2).

    Set verbose @ 1 for basic info on whether training occuring, set @ 2 for print out of tensors.

    Example: w8_lrn_check(pre_train_w8, post_train_w8, verbose=1)
    '''

    pre_train_w8 = pre_train_w8.cpu()
    post_train_w8 = post_train_w8.cpu()

    if loop == 1:
        stage = 'TRAINING STAGE | '
    elif loop == 2:
        stage = 'TEST / VAL STAGE | '

    dims = np.shape(pre_train_w8.cpu())

    output = np.array_equal(pre_train_w8, post_train_w8)

    if verbose == 1:
        if output == True:
            print(stage + 'NO TRAINING OCCURING.')
        elif output == False:
            print(stage + 'TRAINING OCCURING.\n')


    if verbose == 2:
        if output == True:
            print(stage + 'NO TRAINING OCCURING | Pre: {0} | Post: {1}'.format(pre_train_w8[0, 0:5], post_train_w8[0, 0:5]))
        elif output == False:
            print(stage + 'TRAINING OCCURING | Pre: {0} | Post: {1}\n'.format(pre_train_w8[0, 0:5], post_train_w8[0, 0:5]))


def weights_init_kaiming(m):
    '''
    Kaiming Weight Initialization function applied exclusively to the Linear fully connected network layers.

    Takes in a module and applies the specified weight initialization

    Example: net.apply(weights_init_kaiming)

    REF: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch

    '''
    print('-----------------------------')
    print('KAIMING WEIGHT INITIALIZATION')
    classname = m.__class__.__name__
    layer_count = 1
    # for every Linear layer in a model..
    if classname.find('Linear') != -1 or classname.find('Conv2d') != -1:
        'Kaiming Init for ReLU.'
        # https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        'Print Layers in which init applied.'
        layer_count = layer_count + 1
        print('Linear Layer Initialized: ', layer_count)
    print('-----------------------------')


def dwn_resamp(eeg, neo_samps, plotter, verbose):
    '''
    Downsampling to 120Hz as per EEGNet documentation via resampling.

    Assumes: Events x Singleton x Channels x Samples.

    NOTE: plotter argument plots pre and post resampled signals to check for continuity.

    NOTE: verbosity levels 1 and 2, as nums increase so does info on permutations and resampling.

    Example: eeg = dwn_resamp(eeg, neo_samp=120)
    '''
    print('------Down-Sampling')
    from scipy import signal
    n_evnts, n_sing, n_chans, n_samps = np.shape(eeg)
    neo_eeg = np.zeros((n_evnts, neo_samps, n_chans))

    for i in range(n_evnts):
        for j in range(n_chans):
            'Squeeze'
            f = np.squeeze(eeg[i, :, j, :])

            if plotter == 1:
                import matplotlib.pyplot as plt
                plt.figure(1)
                plt.plot(f)
                plt.title('Pre-Resample')

            if verbose == 2:
                print('Squeezed F: ', f.shape)
            'Resamp'
            f = signal.resample(f, neo_samps)

            if plotter == 1:
                import matplotlib.pyplot as plt
                plt.figure(2)
                plt.plot(f)
                plt.title('Post-Resample')
                plt.show()

            if verbose == 2:
                print('Resamp F: ', f.shape)
            'Expand 1'
            f = np.expand_dims(f, axis=0)
            if verbose == 2:
                print('Expand 1 F: ', f.shape)
            'Expand 2'
            f = np.expand_dims(f, axis=0)
            if verbose == 2:
                print('Expand 2 F: ', f.shape)
            'Append'
            neo_eeg[i, :, j] = f

    eeg = np.expand_dims(neo_eeg, axis=1)

    if verbose == 1 or verbose ==2:
        print('Resamp F Signal: ', f.shape)
        print('Neo_EEG: ', neo_eeg.shape)
        print('DWN SAMPED EEG: ', eeg.shape)
    print('\n')

    return eeg


'-------------------------------------------------------------------------------------------------'
'--------------------------------Train / Test Loops-----------------------------------------------'
'-------------------------------------------------------------------------------------------------'

def train(net, trainloader):
    '''
    Training loop for neural network.
    '''
    for epoch in range(5): # no. of epochs
        running_loss = 0
        for data in trainloader:
            # data pixels and labels to GPU if available
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            # set the parameter gradients to zero
            optimizer.zero_grad()
            outputs = net(inputs.float())
            loss = criterion(outputs, labels.long())
            # propagate the loss backward
            loss.backward()
            # update the gradients
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss/len(trainloader)
        print('[Epoch %d] loss: %.3f' %
                      (epoch + 1, epoch_loss))


def test(net, testloader, device, verbose):
    '''
    Validation / Testing loop for neural network.
    '''

    # Assign Variables.
    correct = 0
    total = 0
    all_pred = []
    all_labs = []

    # Print Test Performance Command Line Output.
    print('---Test Performance')

    '---Weights'
    pre_train_w8 = net.fc1.weight.data.clone()
    '---Convert to Net Eval()'
    net.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            outputs = net(inputs.float())
            _, predicted = torch.max(outputs.data, 1)
            '---Accuracy---'
            total += labels.size(0)
            correct += (predicted.cpu() == labels.cpu()).sum().item()
            all_pred = np.append(all_pred, predicted.cpu())
            all_labs = np.append(all_labs, labels.cpu())

    '---Accuracy---'
    test_acc = 100 * correct / total
    from sklearn.metrics import accuracy_score
    test_acc = accuracy_score(all_labs, all_pred) * 100

    if verbose == 1 or verbose == 2:
        print('Test Accuracy: ', test_acc)

    if verbose == 2:
        print('Test NUM all predicted: ', len(all_labs))
        print('Test NUM all labels: ', len(all_pred))
        print('Test NUM total: ', total)
        print('Test NUM correct: ', correct)

    '---Weight Check---'
    post_train_w8 = net.fc1.weight.data.clone()
    if verbose == 1:
        w8_lrn_check(pre_train_w8, post_train_w8, loop=2, verbose=1)
    '---ROC---'
    if verbose == 2:
        print('======Testing / Validation Accuracy')
        print('Test ROC: ', roc_auc_score(all_labs, all_pred))
    '---Recall---'
    if verbose == 2:
        print('Test Recall: ', recall_score(all_labs, all_pred))
    '---Conf Mat---'
    from sklearn.metrics import confusion_matrix
    if verbose == 1 or verbose == 2:
        print('Test Class Confusion Matrix: \n', confusion_matrix(all_labs, all_pred), '\n')
    return test_acc


def real(net, trainloader, valloader, device, optimizer, criterion, num_epochs, live_plot, verbose):
    '''
    Training + Validation / Testing embedded loop for neural network.
    '''

    '---Live Plotting---'
    loss_line = []
    acc_line = []
    v_acc_line = []
    y_v_acc = []
    y_loss = []
    y_acc = []

    '---Weights---'
    # Ensure this is not grabbing from same allocated memory as training tensor
    # therefore do t.clone(), equivalent to np.copy for allocating to new memory location.
    pre_train_w8 = net.fc1.weight.data.clone()

    for epoch in range(num_epochs):
        print('------Training Epoch: ', epoch + 1)
        running_loss = 0
        running_corrects = 0
        for data in trainloader:
            # data pixels and labels to GPU if available
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            # set the parameter gradients to zero
            optimizer.zero_grad()
            outputs = net(inputs.float())
            # REF: https://discuss.pytorch.org/t/interpreting-loss-value/17665/3
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels.long())
            # propagate the loss backward
            loss.backward()
            # update the gradients
            optimizer.step()
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels).item()
        '-------Validation'
        val_acc = test(net, valloader, device, verbose=1)
        '---Performance Metrics---'
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = (running_corrects / len(trainloader.dataset)) * 100

        'Epoch Loss'
        if epoch == 0:
            x_vec = [1]
            y_loss = epoch_loss
            y_acc = epoch_acc
            y_v_acc = val_acc
        elif epoch != 0:
            x_vec = np.arange(epoch + 1)
            y_loss = np.append(y_loss, epoch_loss)
            y_acc = np.append(y_acc, epoch_acc)
            y_v_acc = np.append(y_v_acc, val_acc)

        '-------Live Plotting'
        if live_plot == True:
            loss_line, acc_line, v_acc_line = live_plotter_xy_uber(x_vec, y_loss, y_acc, y_v_acc, loss_line, acc_line, v_acc_line, pause_time=0.01)
        '---Post Epoch Train Performance'
        if verbose == 1:
            print('Epoch Loss: ', epoch_loss)
            print('Epoch Acc: ', epoch_acc, '%')
        '-------Print Weights'
        w8_lrn_check(pre_train_w8, net.fc1.weight.data, loop=1, verbose=1)
    print('Done Training')
    return net


def realEEGNet(net, trainloader, valloader, device, optimizer, criterion, num_epochs, live_plot, verbose):
    '''
    Training + Validation / Testing embedded loop for neural network.
    '''

    '---Live Plotting---'
    loss_line = []
    acc_line = []
    v_acc_line = []
    y_v_acc = []
    y_loss = []
    y_acc = []

    '---Weights---'
    # Ensure this is not grabbing from same allocated memory as training tensor
    # therefore do t.clone(), equivalent to np.copy for allocating to new memory location.
    pre_train_w8 = net.fc1.weight.data.clone()

    for epoch in range(num_epochs):
        print('------Training Epoch: ', epoch + 1)
        running_loss = 0
        running_corrects = 0
        all_pred = []
        all_labs = []
        for data in trainloader:
            # data pixels and labels to GPU if available
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            # set the parameter gradients to zero
            optimizer.zero_grad()
            outputs = net(inputs.float())
            # REF: https://discuss.pytorch.org/t/interpreting-loss-value/17665/3
            _, preds = torch.max(outputs, 1)
            dtype = torch.cuda.FloatTensor
            outputs = outputs.type(dtype)
            loss = criterion(outputs, labels.long())
            # propagate the loss backward
            loss.backward()
            # update the gradients
            optimizer.step()
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.cpu()).item()
            # Conf Mat Variables.
            all_pred = np.append(all_pred, preds.cpu())
            all_labs = np.append(all_labs, labels.cpu())

        '---Performance Metrics---'
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = (running_corrects / len(trainloader.dataset)) * 100
        '---Post Epoch Train Performance'
        if verbose == 1:
            print('Epoch Loss: ', epoch_loss)
            print('Epoch Acc: ', epoch_acc, '%')
        '---Conf Mat---'
        from sklearn.metrics import confusion_matrix
        if verbose == 1 or verbose == 2:
            print('Train Class Confusion Matrix: \n', confusion_matrix(all_labs, all_pred))

        '-------Print Weights'
        w8_lrn_check(pre_train_w8, net.fc1.weight.data, loop=1, verbose=1)

        '-------Validation'
        val_acc = test(net, valloader, device, verbose=1)

        'Epoch Loss'
        if epoch == 0:
            x_vec = [1]
            y_loss = epoch_loss
            y_acc = epoch_acc
            y_v_acc = val_acc
        elif epoch != 0:
            x_vec = np.arange(epoch + 1)
            y_loss = np.append(y_loss, epoch_loss)
            y_acc = np.append(y_acc, epoch_acc)
            y_v_acc = np.append(y_v_acc, val_acc)

        '-------Live Plotting'
        if live_plot == True:
            loss_line, acc_line, v_acc_line = live_plotter_xy_uber(x_vec, y_loss, y_acc, y_v_acc, loss_line, acc_line, v_acc_line, pause_time=0.01)

    print('Done Training')
    return net

'-------------------------------------------------------------------------------------------------'
'-------------------------------------Classes-----------------------------------------------------'
'-------------------------------------------------------------------------------------------------'

'''
Custom Dataset class for DataLoader PyTorch Functionality.
'''
class BRAINDataset(Dataset):
    def __init__(self, eeg, labels=None, transforms=None):
        self.X = eeg
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        'Scaling.'
        from sklearn.preprocessing import MinMaxScaler
        # https://stackabuse.com/implementing-lda-in-python-with-scikit-learn/
        mss = MinMaxScaler(feature_range=[-1, 1])
        # Reshape pre-normalization.
        data = self.X[i, :, :].reshape(-1, 1)
        data = mss.fit_transform(data)
        data = np.squeeze(data)

        'Reshaping.'
        # data = np.expand_dims(self.X[i, :, :], axis=0)
        data = np.expand_dims(data, axis=0)
        data = np.expand_dims(data, axis=1)

        if self.transforms:
            data = self.transforms(data)
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data


'''
Define the neural net class.
'''
class Net(nn.Module):
    '''
    Network notes:

    CONV - ReLU - Dropout - Max Pooling

    https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network
    41

    In the original paper that proposed dropout layers, by Hinton (2012), dropout (with p=0.5)
    was used on each of the fully connected (dense) layers before the output; it was not used on the convolutional
    layers. This became the most commonly used configuration. More recent research has shown some value in applying
    dropout also to convolutional layers, although at much lower levels: p=0.1 or 0.2. Dropout was used after the
    activation function of each convolutional layer: CONV->RELU->DROP.
    '''
    def __init__(self):
        super(Net, self).__init__()
        '------Pod 1'
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20,
                               kernel_size=[1, 250], stride=4)
        self.conv1.add_module('ReLU1', nn.ReLU(inplace=False))
        self.conv1.add_module('BN1', nn.BatchNorm1d(num_features=20))
        self.conv1.add_module('DO1', nn.Dropout(p=0.9, inplace=False))
        self.conv1.add_module('MP1', nn.MaxPool2d(kernel_size=[1, 4]))
        # self.conv1.add_module('AP1', nn.AvgPool2d(kernel_size=[1, 4]))
        '------Pod 2'
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40,
                               kernel_size=[1, 100], stride=4)
        self.conv2.add_module('ReLU2', nn.ReLU(inplace=False))
        self.conv2.add_module('BN2', nn.BatchNorm1d(num_features=40))
        self.conv2.add_module('DO2', nn.Dropout(p=0.9, inplace=False))
        self.conv2.add_module('MP2', nn.MaxPool2d(kernel_size=[1, 4]))
        # self.conv2.add_module('AP2', nn.AvgPool2d(kernel_size=[1, 4]))
        '------Pod 3'
        self.conv3 = nn.Conv2d(in_channels=40, out_channels=60,
                               kernel_size=[1, 50], stride=4)
        self.conv3.add_module('ReLU3', nn.ReLU(inplace=False))
        self.conv3.add_module('BN3', nn.BatchNorm1d(num_features=60))
        self.conv3.add_module('DO3', nn.Dropout(p=0.9, inplace=False))
        self.conv3.add_module('MP3', nn.MaxPool2d(kernel_size=[1, 4]))
        # self.conv3.add_module('AP3', nn.AvgPool2d(kernel_size=[1, 4]))
        '------FC 1'
        self.fc1 = nn.Linear(in_features=4860, out_features=2000)
        self.fc1.add_module('DO4', nn.Dropout(p=0.9, inplace=False))
        '------FC 2'
        self.fc2 = nn.Linear(in_features=2000, out_features=2)

    def forward(self, x):
        printer = False
        tensor_print(x, indicator='Input Dim', printer=printer)
        '---Pod 1'
        x = self.conv1(x)
        tensor_print(x, indicator='Conv 1 Output Dims', printer=printer)
        '---Pod 2'
        x = self.conv2(x)
        tensor_print(x, indicator='Conv 2 Output Dims', printer=printer)
        '---Pod 3'
        x = self.conv3(x)
        tensor_print(x, indicator='Conv 3 Output Dims', printer=printer)
        'Pod Fin'
        x = x.view(x.size(0), -1)
        tensor_print(x, indicator='View Transform Output Dims', printer=printer)
        'Fully Connected 1 + ReLU'
        x = F.relu(self.fc1(x))
        tensor_print(x, indicator='FC 1 Output Dims', printer=printer)
        'Fully Connected 2 + Softmax'
        # REF: https://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/
        x = F.softmax(self.fc2(x), dim=1)
        tensor_print(x, indicator='FC 2 Output Dims', printer=printer)
        return x


'''
EEGNet OG

'''

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 16), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        # OG: self.fc1 = nn.Linear(4*2*7, 1)
        self.fc1 = nn.Linear(2816, 1)


    def forward(self, x):
        printer = True
        tensor_print(x, indicator='Input Dim', printer=printer)
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        tensor_print(x, indicator='1st Layer', printer=printer)
        # OG : x = x.permute(0, 3, 1, 2)
        # x = x.permute(0, 2, 1, 3)
        x = x.permute(0, 3, 1, 2)
        tensor_print(x, indicator='Post Permute', printer=printer)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        tensor_print(x, indicator='2nd Layer', printer=printer)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        tensor_print(x, indicator='3rd Layer', printer=printer)

        # FC Layer
        # OG: x = x.view(-1, 4*2*7)
        x = x.view(-1, 2816)
        tensor_print(x, indicator='Post Reshape', printer=printer)
        x = F.sigmoid(self.fc1(x))
        tensor_print(x, indicator='Fully Con', printer=printer)
        return x

'''
Plot TENSOR: https://discuss.pytorch.org/t/understanding-deep-network-visualize-weights/2060/6

import torch
import torchvision.models as models
from matplotlib import pyplot as plt

def plot_kernels(tensor, num_cols=6):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

'''
