#TODO : mentionner auteur

import numpy as np
from sklearn.metrics import pairwise_distances
from scipy import stats
import pdb
# from copy import copy as copy
from copy import deepcopy as deepcopy

from cardinal.random import RandomSampler
from cardinal.zhdanov2019 import TwoStepKMeansSampler

# from main_fedrep import update_user_model   #TODO changer importation croisée
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


# kmeans ++ initialization
def init_centers(X, K, print_distances=False):
    index = np.argmax([np.linalg.norm(s, 2) for s in X])  #Sélection du premier embedding possédant la plus grande uncertainty
    mu = [X[index]]
    indsAll = [index]
    centInds = [0.] * len(X)
    cent = 0                    # ?
    if print_distances: print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        if print_distances: print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


def get_gradients_embeddings(model, X_train, nb_class, embedding_dim):
    """ 
    BADGE selection purpose only
    We only compute gradients embeddings of unlabeled samples
    """
    # embedding_dim = self.model.get_embedding_dim()
    embedding = np.zeros([len(X_train), embedding_dim * nb_class])   # Embedding for all unlabeled samples
    out, softmax = model.get_outputs(X_train)    #TODO : manage corresponding function
    max_indices = np.argmax(softmax,1)
    for sample_id in range(len(X_train)):
        for target_class in range(nb_class):
            if target_class == max_indices[sample_id]:
                embedding[sample_id][embedding_dim * target_class : embedding_dim * (target_class+1)] = deepcopy(out[sample_id]) * (1 - softmax[sample_id][target_class])
            else:
                embedding[sample_id][embedding_dim * target_class : embedding_dim * (target_class+1)] = deepcopy(out[sample_id]) * (-1 * softmax[sample_id][target_class])
    return embedding
# def get_gradients_embeddings(model, X_train, y_train, labeled_mask, nb_class, embedding_dim):
#     """ 
#     BADGE selection purpose only
#     We only compute gradients embeddings of unlabeled samples
#     """
#     # embedding_dim = self.model.get_embedding_dim()
#     embedding = np.zeros([len(y_train[~labeled_mask]), embedding_dim * nb_class])   # Embedding for all unlabeled samples
#     out, softmax = model.get_outputs(X_train[~labeled_mask])    #TODO : manage corresponding function
#     max_indices = np.argmax(softmax,1)
#     for sample_id in range(len(y_train[~labeled_mask])):
#         for target_class in range(nb_class):
#             if target_class == max_indices[sample_id]:
#                 embedding[sample_id][embedding_dim * target_class : embedding_dim * (target_class+1)] = deepcopy(out[sample_id]) * (1 - softmax[sample_id][target_class])
#             else:
#                 embedding[sample_id][embedding_dim * target_class : embedding_dim * (target_class+1)] = deepcopy(out[sample_id]) * (-1 * softmax[sample_id][target_class])
#     return embedding


#TODO update sampler so that it corresponds to another sampler (no code distinction in 'active_learning' function)
class BadgeSampler:
    def __init__(self):
        return
    
    # def fit(self, X_train=None, sampling_batch_size=None):
    #     # self.nb_samples = len(X_train)
    #     # if sampling_batch_size is None: self.sampling_batch_size = 0.01*self.nb_samples
    #     # else: self.sampling_batch_size = sampling_batch_size
    #     return self

    def select_samples(self, model, X_train, mask, sampling_batch_size=None):
        """ Return : Selected sample indices among all train dataset indices """
        #TODO : get embeddding dim a partir du modele
        #TODO : get nb_class a partir du modele #ATTNETION SI MULTI TASK les tete n'auront pas forcement la meme taille de clf
        nb_samples = len(X_train)
        if sampling_batch_size is None: 
            sampling_batch_size = 0.01*nb_samples

        grad_emb = get_gradients_embeddings(model, X_train[~mask], nb_class, embedding_dim)
        idxs_unlabeled = np.arange(self.nb_samples)[~mask]
        chosen = init_centers(grad_emb, n=sampling_batch_size)
        return idxs_unlabeled[chosen]


SAMPLERS = {
    'iwkmeans': TwoStepKMeansSampler,
    'random': RandomSampler,
    'badge': BadgeSampler,
    # 'entropy':
}

def get_sampler(sampler_name):
    '''
    Return corresponding sampler class 
    Names available : 'badge', 'random', 'entropy', 'iwkmeans'
    '''
    return SAMPLERS[sampler_name]


def active_learning_selection(args, sampler_name, X_train, y_train, mask, idxs_users, net_glob, w_locals, w_glob_keys, fal_global):
    """
    #TODO AL with torch
    #TODO X_train = ?
    #TODO copy paste badge torch code
    Active learning selection
    Updates the dataset mask corresponding to the selected samples
    """
    sampler = get_sampler(sampler_name)

    #TODO : re-fit le sampler sur les données de chaque client ?
    #TODO revoir initialisations samplers
    # TODO essayer de faire coincider les argumnents des methodes non badge et badge
    if sampler is typeof(BadgeSampler):
        if fal_global:
            sampler = sampler()
            # TODO : train a new head for global sampling ?
            # Sampling 
            selected = sampler.select_samples(X_train, mask)#model, X_train, mask
            mask[selected] = True

            # Reconstruct right mask shape
            # mask[indices[~mask][selected]] = True
        else:
            for idx in idxs_users:
                net_local = update_user_model(args, net_glob, idx, w_locals, w_glob_keys)
                user_mask = mask[idx]
                X, y = 0, 0
                sampler = sampler()
                #Sampling
                selected = sampler.select_samples(net_local, X_train, user_mask)
                mask[idx][np.arange(X_train.shape[0])[~user_mask][selected]] = True
                mask[selected] = True

    else:
        if fal_global:
            sampler = sampler()
            # Sampling 
            sampler.fit(X_train[mask], y_train[mask])
            selected = sampler.select_samples(X_train[~mask])

            # Reconstruct right mask shape
            # mask[indices[~mask][selected]] = True
        else:
            for idx in idxs_users:
                net_local = update_user_model(args, net_glob, idx, w_locals, w_glob_keys)
                user_mask = mask[idx]
                sampler = sampler()
                X, y = 0, 0
                sampler.fit(X[user_mask], y[user_mask])
                selected = sampler.select_samples(X[~user_mask])
                mask[idx][np.arange(X.shape[0])[~user_mask][selected]] = True


    # AL selection
    # if sampler_name == "BADGE":
    #     selected = sampler.select_samples(
    #         grad_embeddings = badge.get_gradients_embeddings(),
    #         labeled_mask = mask)
    #     mask[selected] = True
        
    # else:
    #     sampler.fit(X_train[mask], y_train[mask])
    #     selected = sampler.select_samples(X_train[~mask])
    #     mask[indices[~mask][selected]] = True
    return mask