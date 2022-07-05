# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# This program implements FedRep under the specification --alg fedrep, as well as Fed-Per (--alg fedper), LG-FedAvg (--alg lg), 
# FedAvg (--alg fedavg) and FedProx (--alg prox)

# python main_fedrep.py --alg fedrep --dataset cifar10 --num_users 20
import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data
from models.Update import LocalUpdate
from models.test import test_img_local_all

from utils.active_learning import active_learning_selection, get_sampler
from utils.functions import Experiment, save_results

import time

def load_dataset(args):
    lens = np.ones(args.num_users)

    if 'cifar' in args.dataset or args.dataset == 'mnist':
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])

        clients = None

    else:
        if 'femnist' in args.dataset:
            train_path = './leaf-master/data/' + args.dataset + '/data/mytrain'
            test_path = './leaf-master/data/' + args.dataset + '/data/mytest'
        else:
            train_path = './leaf-master/data/' + args.dataset + '/data/train'
            test_path = './leaf-master/data/' + args.dataset + '/data/test'
        clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
        lens = []

        for iii, c in enumerate(clients):
            lens.append(len(dataset_train[c]['x']))
        dict_users_train = list(dataset_train.keys()) 
        dict_users_test = list(dataset_test.keys())
        # print("lens", lens)
        # print(clients)

        for c in dataset_train.keys():
            dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
            dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))

    return dataset_train, dataset_test, dict_users_train, dict_users_test, lens, clients

def build_model(args):
    net_glob = get_model(args)
    net_glob.train()
    if args.load_fed != 'n':    # TODO .load_fed not available -> path to fed model previously trained
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))
    return net_glob

def get_representation_parameters(args, net_glob):
    """
    Specify the representation parameters (in w_glob_keys) and head parameters (all others)
    """
    total_num_layers = len(net_glob.state_dict().keys())
    # print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    if args.alg == 'fedrep' or args.alg == 'fedper':
        if 'cifar' in  args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,2,3]] # I replaced [0,1,3,4] by [0,1,2,3] (list index out of range)                     #TODO : difference between net_glob.weight_keys et net_keys
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,2]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0,1,2,3,4,5]]
        else:
            w_glob_keys = net_keys[:-2]
    elif args.alg == 'lg':
        if 'cifar' in  args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [1,2]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [2,3]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0,6,7]]
        else:
            w_glob_keys = net_keys[total_num_layers - 2:]

    if args.alg == 'fedavg' or args.alg == 'prox' or args.alg == 'maml':
        w_glob_keys = []
    if 'sent140' not in args.dataset:
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    # print(total_num_layers)
    # print(w_glob_keys)
    # print(net_keys)

    return w_glob_keys

def generate_users_models(args,net_glob):
    """
    Generate list of local models for each user
    """
    w_locals = {}   # Dictionnary of dictionnaries of users models weights
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] =net_glob.state_dict()[key]
        w_locals[user] = w_local_dict
    return w_locals

def get_local_update_class(args, dataset_train, idx, dict_users_train, indd):
    """
    Get the generic local update class for current updated user and iteration
    """
    if 'femnist' in args.dataset or 'sent140' in args.dataset:
        if args.epochs == iter:
            local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_ft]], idxs=dict_users_train, indd=indd)    # TODO m_ft arg attribute unavailable    # TODO : useful to discriminate args.epochs == iter or not here ?? -> args.m_ft or args.m_tr
        else:
            local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_tr]], idxs=dict_users_train, indd=indd)    # TODO m_tr arg attribute unavailable
    else:
        if args.epochs == iter:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft])
        else:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr])

    return local

def update_user_model(args, net_glob, user_idx, w_locals, w_glob_keys):
    """
    Update local user model (net_local) before local training
    """
    net_local = copy.deepcopy(net_glob)
    w_local = net_local.state_dict()

    # Augmentation of local model (with some w_locals[idx] weights) - Adding heads      #TODO verif     #w_locals conserve les updates des users heads ?
    if args.alg != 'fedavg' and args.alg != 'prox':
        for k in w_locals[user_idx].keys():          #TODO : pourquoi pas regarder k dans w_local.keys() ?
            if k not in w_glob_keys:
                w_local[k] = w_locals[user_idx][k]
    net_local.load_state_dict(w_local)
    return net_local

def update_global_model(net_glob, w_glob):
    """ Update the global network """
    net_glob.load_state_dict(w_glob)
    return net_glob

def train_user(args, w_glob, net_glob, idx, w_locals, w_glob_keys, dataset_train, dict_users_train, indd, clients, last):
    start_in = time.time()
    local = get_local_update_class(args, dataset_train, idx, dict_users_train, indd)

    # Update local user model before local training
    net_local = update_user_model(args, net_glob, idx, w_locals, w_glob_keys)

    # Local training
    #TODO : find a way to don't instanciate this everysingle user of each single epoch (memory problems ?)
    if 'femnist' in args.dataset or 'sent140' in args.dataset:
        w_local, loss, indd = local.train(net=net_local.to(args.device), ind=idx, idx=clients[idx], w_glob_keys=w_glob_keys, lr=args.lr,last=last)
    else:
        w_local, loss, indd = local.train(net=net_local.to(args.device), idx=idx, w_glob_keys=w_glob_keys, lr=args.lr, last=last)
    loss = copy.deepcopy(loss)
    # loss_locals.append(copy.deepcopy(loss))
    # total_len += lens[idx]

    # Active learning 
    #TODO mettre en argument fonction train_user
    # mask = active_learning_selection(net_local, sampler, mask)
    #TODO return mask

    # Sending weihts to the server
    # Adding all local weights to the global weights after each user local training, before Fed Avg (this sum is weighted by the user weight "lens[idx]")
    if len(w_glob) == 0:     
        w_glob = copy.deepcopy(w_local)
        for k, key in enumerate(net_glob.state_dict().keys()):
            w_glob[key] = w_glob[key]*lens[idx] # 
            w_locals[idx][key] = w_local[key]
    else:
        for k, key in enumerate(net_glob.state_dict().keys()):
            if key in w_glob_keys:                          #TODO : why discriminate if key in w_glob_keys
                w_glob[key] += w_local[key]*lens[idx]
            else:
                w_glob[key] += w_local[key]*lens[idx]   #w_glob[key] = w_local[key]*lens[idx]   #TODO (equal sign) ?
            w_locals[idx][key] = w_local[key]
    
    train_time = time.time() - start_in

    return w_glob, loss, train_time, indd

def fed_avg(net_glob, w_glob, total_len):
    """
    Get weighted average for global weights
    """
    for k in net_glob.state_dict().keys():      #TODO difference between w_glob_keys and net_glob.state_dict().keys()
        w_glob[k] = torch.div(w_glob[k], total_len)
    return w_glob


"""
#TODO
- FAL setting
- Centralized setting
- Aggregation gradients
- add seed to noniid function (revoir nonidd fct) (need to see how the dataset is shuffled in "load dataset" also) + add new non idd method ?
- diff entre clients et idxs_users
- Attention formats acc lorsque retour de multiples client accuracies
- Regarder si pas déja des resultats d'experiments similaires avant d'écraser la save
-
"""


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # args.num_classes = len(np.unique(y_train))    #TODO

    # Load dataset
    dataset_train, dataset_test, dict_users_train, dict_users_test, lens, clients = load_dataset(args)
    #len looks like users weights (like relevance)  TODO verify

    # OTHER PARAMETERS
    exp_args = Experiment(is_AL=True, is_fal_global=False, individual_results=True, sampler="BADGE")

    # build model
    net_glob = build_model(args)

    w_glob_keys = get_representation_parameters(args, net_glob)

    # Print percentage of global params compared to local params
    if args.alg == 'fedrep' or args.alg == 'fedper' or args.alg == 'lg':
        num_param_glob = 0
        num_param_local = 0
        for key in net_glob.state_dict().keys():
            num_param_local += net_glob.state_dict()[key].numel()
            # print(num_param_local)
            if key in w_glob_keys:
                num_param_glob += net_glob.state_dict()[key].numel()
        percentage_param = 100 * float(num_param_glob) / num_param_local
        print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
            num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    print("learning rate, batch size: {}, {}".format(args.lr, args.local_bs))

    # Generate list of local models for each user
    w_locals = generate_users_models(args,net_glob)

    # TRAINING

    indd = None     # indices of embedding for sent140
    loss_train = [] # List of epochs losses
    accs = []
    times = []  # Cumulated sum list of maximum user local trining times (the value at one FL epoch is the sum of maximum user local training times from previous epochs)
    accs10 = 0
    accs10_glob = 0
    start = time.time()
    for iter in range(args.epochs+1):
        w_glob = {}
        loss_locals = []    # List of local loss of each client after local training epochs
        last = iter == args.epochs

        # Selection of users who will participate to the training for this epoch
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        times_in = []   # Local training times of each user
        total_len=0     # Sum of user's weights used for current training epoch

        for ind, idx in enumerate(idxs_users):
            # Local training
            w_glob, loss, train_time, indd = train_user(args, w_glob, net_glob, idx, w_locals, w_glob_keys, dataset_train, dict_users_train, indd, clients, last)

            user_weight = lens[idx]
            total_len += user_weight
            loss_locals.append(loss)
            times_in.append(train_time)
        loss_avg = sum(loss_locals) / len(loss_locals)  # Mean loss over clients local trains
        loss_train.append(loss_avg)

        # Get weighted average for global weights
        w_glob = fed_avg(net_glob, w_glob, total_len)

        # Active Learning 
        if exp_args.is_AL:
            mask = active_learning_selection(exp_args.sampler_name, mask, idxs_users, exp_args.is_fal_global)

        # TODO comprendre les 3 lignes suivantes : Est ce pertinent de re-update le modèle local après l'aggregation globale ? Car je crois qu'on en fait rien de w_local avant la prochaine iteration
        w_local = net_glob.state_dict()     #TODO : pourquoi copier les poids du modèle global pour le modèle local ?? La tete du modèle va etre affectée non ?
        for k in w_glob.keys():
            w_local[k] = w_glob[k]

        # Update the global network
        if args.epochs != iter:
            net_glob = update_global_model(net_glob, w_glob)

        # Prints + some metrics
        if iter % args.test_freq==args.test_freq-1 or iter>=args.epochs-10:

            #Update computing times
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))

            # Evaluation
            acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                        w_glob_keys=w_glob_keys, w_locals=w_locals,indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=exp_args.get_individual_results)
            accs.append(acc_test)

            # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
            if iter != args.epochs:
                print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(iter, loss_avg, loss_test, acc_test))
            else:
                # in the final round, we sample all users, and for the algs which learn a single global model, we fine-tune the head for 10 local epochs for fair comparison with FedRep
                print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                        loss_avg, loss_test, acc_test))
            if iter >= args.epochs-10 and iter != args.epochs:
                accs10 += acc_test/10

            # below prints the global accuracy of the single global model for the relevant algs
            if args.alg == 'fedavg' or args.alg == 'prox':
                acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                        w_locals=None,indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False)
                if iter != args.epochs:
                    print('Round {:3d}, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                        iter, loss_avg, loss_test, acc_test))
                else:
                    print('Final Round, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                        loss_avg, loss_test, acc_test))
            if iter >= args.epochs-10 and iter != args.epochs:
                accs10_glob += acc_test/10

        if iter % args.save_every==args.save_every-1:   #TODO : save energy not available
            model_save_path = './save/accs_'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) +'_'+ str(args.shard_per_user) +'_iter' + str(iter)+ '.pt'
            torch.save(net_glob.state_dict(), model_save_path)

    print('Average accuracy final 10 rounds: {}'.format(accs10))
    if args.alg == 'fedavg' or args.alg == 'prox':
        print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.time()
    # print(end-start)
    # print(times)
    # print(accs)

    #Saving accuracies 1
    base_dir = './save/accs_' + args.alg + '_' +  args.dataset + str(args.num_users) +'_'+ str(args.shard_per_user) + '.csv'
    # user_save_path = base_dir     # user_save_path unused
    accs = pd.DataFrame(np.array(accs), columns=['accs'])
    accs.to_csv(base_dir, index=False)

    # Saving accuracies 2

    # save_results(experiments, metric_value='', directory="FAL", dataset=args.dataset, results_folder=results_folder, clientdata_split_type="non-iid", nb_clients=nb_clients, train_epochs=train_epochs)
    # save_results(experiments, metric_value, directory, dataset, results_folder, clientdata_split_type, nb_clients, train_epochs, personalisation)
    # exp


def save_res(accuracies, exp_args):
    #TODO directory management
    if exp_args.get_individual_results:
        base_dir = './save/accs_' + args.alg + '_' +  args.dataset + str(args.num_users) +'_indiv_'+ str(args.shard_per_user) + '.csv'
        # data_to_save = []

        # individual_results = True
        # for round_id, iteration_accuracies in enumerate(accuracies):
            # data_to_save.append()
        df = pd.DataFrame([
            [
                {'setting': exp_args.name, 'sampler': exp_args.sampler_name, 'client':int(client_id), 'round': int(round_id), 'accuracy': acc} for client_id, acc in enumerate(iteration_accuracies)
            ] for round_id, iteration_accuracies in enumerate(accuracies)])
            # df = pd.concat([df, new_df], ignore_index=True)

    else:
        base_dir = './save/accs_' + args.alg + '_' +  args.dataset + str(args.num_users) +'_global_'+ str(args.shard_per_user) + '.csv'
        # user_save_path = base_dir     # user_save_path unused
        df = pd.DataFrame(np.array(accuracies), columns=['accs'])
    
    df.to_csv(base_dir, index=False)