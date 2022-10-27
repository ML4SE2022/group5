import copy
import torch
import numpy as np
from annoy import AnnoyIndex
from collections import defaultdict
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
KNN_TREE_SIZE = 20
DISTANCE_METRIC = 'euclidean'

def create_type_space(custom_model, inputs, m_labels, labels):
    """
    Creates the type space based on the inputs and their corresponding labels
    """
    
    # Make sure imputs are labeled
    assert len(inputs) == len(m_labels)
    
    # Cache the type space mappings
    computed_mapped_batches_train = []
    computed_mapped_labels_train = []
    with torch.no_grad():
        
        annoy_idx = AnnoyIndex(8, DISTANCE_METRIC)
        count = 0
        
        # Iterate through the data set
        for i, (inp, m_label, label) in tqdm(enumerate(zip(inputs, m_labels, labels)), total=len(inputs), desc="Create type space"):
            
            # Get the type space mapping from the model
            output = custom_model(input_ids=torch.cat((inp, m_label), 0))
                        
            # Cache the mapping of the masked token only
            computed_mapped_batches_train.append(output)
            computed_mapped_labels_train.append(label)
        
            if i > 0 and i % 1000 == 0:
                annoy_index = create_knn_index(annoy_idx, computed_mapped_batches_train, None, computed_mapped_batches_train[0].shape[0], count)
                count += 1
        
        annoy_index = create_knn_index(annoy_idx, computed_mapped_batches_train, None, computed_mapped_batches_train[0].shape[0], count)
        annoy_idx.build(KNN_TREE_SIZE)

    return annoy_index, computed_mapped_labels_train

def create_knn_index(annoy_idx: AnnoyIndex, train_types_embed: np.array, valid_types_embed: np.array, type_embed_dim:int, count: int) -> AnnoyIndex:
    """
    Creates KNNs index for given type embedding vectors
    """
    
    for i, v in enumerate(tqdm(train_types_embed, total=len(train_types_embed), desc="KNN index")):
        annoy_idx.add_item(i, v)
        
    annoy_idx.build(KNN_TREE_SIZE)
    annoy_idx.save("type-model/space_intermediary" + str(count) + ".ann")

    # TODO: add valid type embeddings to space
    annoy_idx.unload()
    annoy_idx.unbuild()

    return annoy_idx

def map_type(custom_model, inputs, m_labels):
    """
    Maps an input to the type space
    """
    with torch.no_grad():
        computed_embed_batches_test = []
        
        for inp, m_label in tqdm(zip(inputs, m_labels), total=len(inputs), desc="Map type"):
            
            # Get the type space mapping from the model
            output = custom_model(input_ids=torch.cat((inp, m_label), 0))
            
            # Cache the mapping of the masked token only
            computed_embed_batches_test.append(output)
        
        return computed_embed_batches_test

def predict_type(types_embed_array: np.array, types_embed_labels: np.array, indexed_knn: AnnoyIndex, k: int):
    """
    Predict type of given type embedding vectors
    """

    pred_types_embed = []
    pred_types_score = []
    for i, embed_vec in enumerate(tqdm(types_embed_array, total=len(types_embed_array), desc="Finding KNNs & Prediction")):
        
        # Get the distances to the KNN
        idx, dist = indexed_knn.get_nns_by_vector(embed_vec, k, include_distances=True)
        
        # Compute the scores according to the formula
        pred_idx_scores = compute_types_score(dist, idx, types_embed_labels)
        
        # Cache the scores and the labels
        pred_types_embed.append([i for (i, s) in pred_idx_scores])
        pred_types_score.append(pred_idx_scores)
    
    return pred_types_embed, pred_types_score

def compute_types_score(types_dist: list, types_idx: list, types_embed_labels: np.array):
        types_dist = 1 / (np.array(types_dist) + 1e-10) ** 2
        types_dist /= np.sum(types_dist)
        types_score = defaultdict(int)
        for n, d in zip(types_idx, types_dist):
            types_score[types_embed_labels[n]] += d
        
        return sorted({t: s for t, s in types_score.items()}.items(), key=lambda kv: kv[1], reverse=True)
