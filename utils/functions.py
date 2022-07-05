import os
import numpy as np
import matplotlib.pyplot as plt
from cardinal.plotting import plot_confidence_interval
import pandas as pd
from tensorflow.keras.datasets import mnist, cifar10
# from tensorflow_federated.simulation.datasets import celeba
# import hub


# TODO : revoir nomenclature metric_values car c'est quasiment toujours ne nombre d'epochs


class Experiment:
    """
    sampler : 'iwkmeans' or 'BADGE'
    fusion_method : 'FedAvg' or 'gradients
    """
    def __init__(self, name, is_FL, sampler=None, is_global_FAL=False, fusion_method='gradients', one_by_one_gradients=True, individual_results=False):
        self.name = name
        self.accuracies = []
        self.is_FL = is_FL
        self.sampler = sampler
        self.individual_results = individual_results
        if is_FL:
            self.is_global_FAL = is_global_FAL
            # self.fusion_method = fusion_method
            # self.one_by_one_gradients = one_by_one_gradients

            #TODO useful ?
            # if fusion_method == 'FedAvg':
            #     assert is_global_FAL == False
            #     self.one_by_one_gradients = False # assert one_by_one_gradients == False
                
            # if sampler == 'BADGE':
            #     assert fusion_method == "gradients"

        else:
            self.individual_results = False
            self.is_global_FAL = None
            # self.fusion_method = None
            # self.one_by_one_gradients = None
        return
    
    def add_result(self, accuracies, individual_results=False):
        """
        if individual_results:
            [
                fold1[
                        client1[iteration accuracies]
                        ...
                    ]
                ...
            ]
        else:
            [
                fold1[iteration accuracies]
                ...
            ]
        """
        self.accuracies.append(accuracies)
        return


def directory_management(directory, dataset, results_folder, clientdata_split_type, personalisation):
    """
    Instructions:

    directory should be in ['FL', 'FAL', 'AFL']
    dataset : 'mnist' or 'cifar10'
    results_folder should be in ['AL scale','round', 'epochs', 'sampler', 'other']
    """

    # assert results_folder in ['AL scale', 'round', 'epochs', 'sampler', 'other'], f"results_folder should be in ['AL scale','round', 'epochs', 'sampler', 'other'] but got {results_folder} instead"
    assert directory in ['FL', 'FAL', 'AFL'], f"directory should be in ['FL', 'FAL', 'AFL'] but got {directory} instead"
    assert dataset in ['mnist']

    script_dir = os.path.dirname(f"{directory}.py")  #__file__
    if personalisation:
        results_dir = os.path.join(script_dir, f'results/{directory}/{dataset}/{clientdata_split_type}-data-split/personalisation/{results_folder}/')
    else :
        results_dir = os.path.join(script_dir, f'results/{directory}/{dataset}/{clientdata_split_type}-data-split/no-personalisation/{results_folder}/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    return results_dir


#TODO : check data class
def save_results(experiments, metric_value, directory, dataset, results_folder, clientdata_split_type, nb_clients, train_epochs, personalisation):

    results_dir = directory_management(directory, dataset, results_folder, clientdata_split_type, personalisation)

    # Create results dataframe
    individual_results = False
    df = None
    for exp in experiments:
        for n_fold, one_fold_exp_accuracies in enumerate(exp.accuracies):    #for each "seed"

            if exp.individual_results:
                individual_results = True
                for client_id, client_one_fold_exp_accuracies in enumerate(one_fold_exp_accuracies):
                    new_df = pd.DataFrame([{'setting': exp.name, 'sampler': exp.sampler, 'client':int(client_id), 'n_fold': int(n_fold), 'round': int(round), 'accuracy': acc} for round, acc in enumerate(client_one_fold_exp_accuracies)])
                    df = pd.concat([df, new_df], ignore_index=True)
            else:
                new_df = pd.DataFrame([{'setting': exp.name, 'sampler': exp.sampler, 'n_fold': int(n_fold), 'round': int(round), 'accuracy': acc} for round, acc in enumerate(one_fold_exp_accuracies)])
                df = pd.concat([df, new_df], ignore_index=True)

    # Save results dataframe
    # TODO Regarder si pas déja des resultats d'experiments similaires avant d'écraser la save
    if personalisation:
        if individual_results:
            df.to_csv(results_dir + 'results_indiv_{}_{}.csv'.format(metric_value, results_folder), index=False)
        else:
            df.to_csv(results_dir + 'results_global_{}_{}.csv'.format(metric_value, results_folder), index=False)
    else:
        df.to_csv(results_dir + 'results_{}_{}.csv'.format(metric_value, results_folder), index=False)

    _plot_results(experiments, metric_value, train_epochs, nb_clients, directory, dataset, results_folder, personalisation, individual_results, clientdata_split_type, save=1)
    return


#TODO : more detailed plot with settings (get sampler)
def _plot_results(experiments, metric_value, train_epochs, nb_clients, directory, dataset, results_folder, clientdata_split_type, personalisation, individual_results, begin_index=0, save=0):  #all_accuracies_fed, all_accuracies_fed_grad, all_accuracies_centralized, all_accuracies_fed_grad_global,
    """
    Plot experiments results

    Inputs:
    - metric name : the metric we are analysing (the one we fixed during the different experiments)
    - metric value : fixed value of this metric
    - train_epochs : number of epochs for each client of the FL network
    - directory : name of the FLxAL setting directory we are using ('FL', 'FAL' or 'AFL')
    - begin_index : index of the first FL iteration to plot
    - save : boolean for saving the plots in the given directory folder
    """

    # HARD CODED PARAMETER
    compute_mean_over_clients = True


    # Directory management
    if save:
        results_dir = directory_management(directory, dataset, results_folder, clientdata_split_type, personalisation)

    if compute_mean_over_clients: assert individual_results
    n_splits = np.array(experiments[0].accuracies).shape[0]
    n_rounds = np.array(experiments[0].accuracies).shape[-1]    #Take into account different results shapes according to the parameter "individual results"

    # Compute mean accuracy over seeds experiments
    MEAN_ALL_EXP_ACC = []
    for exp in experiments:
        exp_acc = exp.accuracies
        if exp.individual_results:
            nb_clients = len(exp.accuracies[0])
            clients_mean_exp_acc = []

            for client_id in range(nb_clients):
                # Mean accuracy over folds
                client_mean_exp_acc = np.sum(np.array(exp_acc)[:, client_id], axis=0) / n_splits
                client_mean_exp_acc = np.reshape(client_mean_exp_acc,(client_mean_exp_acc.shape[0],))
                clients_mean_exp_acc.append(client_mean_exp_acc)

            # Mean acuracy over clients
            if compute_mean_over_clients:
                mean_clients_mean_exp_acc = np.sum(np.array(clients_mean_exp_acc), axis=0) / nb_clients
                clients_mean_exp_acc.append(mean_clients_mean_exp_acc) #Adding the mean of the clients accuracies at the end of the list of clients_mean_accuracies

            MEAN_ALL_EXP_ACC.append(clients_mean_exp_acc)
        else:
            mean_exp_acc = np.sum(np.array(exp_acc), axis=0) / n_splits
            mean_exp_acc = np.reshape(mean_exp_acc,(mean_exp_acc.shape[0],))
            MEAN_ALL_EXP_ACC.append(mean_exp_acc)
        # print('[SHAPE] {} \t {} \t {}'.format(exp_name, (len(exp_acc),len(exp_acc[0])), mean_exp_accuracies.shape))

    x = np.arange(begin_index, n_rounds)

    # Plot mean accuracies
    plt.figure(figsize=(15,10))
    for mean_exp_acc, exp in zip(MEAN_ALL_EXP_ACC, experiments):
        exp_name = exp.name
        try:
            if exp.individual_results:
                for client_id, client_mean_exp_acc in enumerate(mean_exp_acc):
                    if (client_id==nb_clients) & compute_mean_over_clients:
                        # Plot mean clients accuracies
                        plt.plot(x, client_mean_exp_acc[begin_index:], label=f'{exp_name} - mean clients')
                    else:
                        plt.plot(x, client_mean_exp_acc[begin_index:], label=f'{exp_name} - client{client_id}')
            else:
                plt.plot(x, mean_exp_acc[begin_index:], label=f'{exp_name}')
        except:
            print(f"Problem plotting the result of {exp_name} experiment")
    plt.title(f'Model accuracy  |  Settings : {nb_clients}clients, {train_epochs}epochs, {n_splits}folds, {experiments[0].sampler} sampler', fontsize = 20)
    plt.ylabel('accuracy')
    plt.xlabel('FL rounds')
    plt.legend()
    if save: 
        plt.savefig(results_dir + f'{metric_value}_accuracy' + ('_indiv' if individual_results else '_global') if personalisation else '' + '.jpg')

    # Confidence intervals
    plt.figure(figsize=(15,10))
    for exp_id, exp in enumerate(experiments):
        if exp.individual_results:
            for client_id in range(nb_clients):
                if (client_id==nb_clients) & compute_mean_over_clients:
                    # plot_confidence_interval(x, np.array(exp.accuracies)[:, client_id, begin_index:], label=f'{exp.name} - mean clients')
                    plt.plot(x, MEAN_ALL_EXP_ACC[exp_id][-1][begin_index:], label=f'{exp_name} - mean clients')
                else:
                    plot_confidence_interval(x, np.array(exp.accuracies)[:, client_id, begin_index:], label=f'{exp.name} - client{client_id}')
        else:
            plot_confidence_interval(x, np.array(exp.accuracies)[:,begin_index:], label=f'{exp.name}')
    plt.title(f'Model accuracy  |  Settings : {nb_clients}clients, {train_epochs}epochs, {n_splits}folds, {experiments[0].sampler} sampler', fontsize = 20)
    plt.ylabel('accuracy')
    plt.xlabel('FL rounds')
    plt.legend()
    if save: 
        plt.savefig(results_dir + f'{metric_value}_confidence-accuracy' + ('_indiv' if individual_results else '_global') if personalisation else '' + '.jpg')

    plt.show()
