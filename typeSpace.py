

def create_type_space(inputs=input_list[:4], labels=labels[:4]):
    """
    Creates the type space based on the inputs and their corresponding labels
    """
    
    # Make sure imputs are labeled
    assert len(inputs) == len(labels)
    
    # Cache the type space mappings
    computed_mapped_batches_train = []
    with torch.no_grad():
        
        # Iterate through the data set
        for inp, label in zip(inputs, labels):
            
            # Tokenize the code
            nl_tokens = tokenizer.tokenize("")
            code_tokens = tokenizer.tokenize(inp)
            tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
            tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
            
            # Get the type space mapping from the model
            output = custom_model.forward(torch.tensor(tokens_ids)[None,:])
            
            # Select masked tokens
            masked_tokens = [c for c, token in enumerate(code_tokens) if token == "<mask>"]
            
            print(masked_tokens)
            
            # For this version, assume only one mask
            assert len(masked_tokens) == 1
            
            # Selected only the masked tokens from the output
            vals = output.logits.cpu().numpy()
            predicted_masks = [vals[0][i] for i in masked_tokens]
            
            # Cache the mapping of the masked token only
            computed_mapped_batches_train.append(predicted_masks)
        
        # Create the type space
        annoy_index = create_knn_index(computed_mapped_batches_train, None, computed_mapped_batches_train[0][0].size)
    return annoy_index

def create_knn_index(train_types_embed: np.array, valid_types_embed: np.array, type_embed_dim:int) -> AnnoyIndex:
    """
    Creates KNNs index for given type embedding vectors, taken from Type4Py
    """
    
    annoy_idx = AnnoyIndex(type_embed_dim, DISTANCE_METRIC)

    for i, v in enumerate(tqdm(train_types_embed, total=len(train_types_embed), desc="KNN index")):
        print(v[0])
        annoy_idx.add_item(i, v[0])

    annoy_idx.build(KNN_TREE_SIZE)
    return annoy_idx

def map_type(inputs=input_list[:4]):
    """
    Maps an input to the type space
    """
    with torch.no_grad():
        computed_embed_batches_test = []
        computed_embed_labels_test = []
        
        for inp in inputs:

            # Tokenize the code
            nl_tokens = tokenizer.tokenize("")
            code_tokens = tokenizer.tokenize(inp)
            tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
            tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
            
            # Get the type space mapping from the model
            output = custom_model.forward(torch.tensor(tokens_ids)[None,:])
            
            # Select masked tokens
            masked_tokens = [c for c, token in enumerate(code_tokens) if token == "<mask>"]
            
            # For this version, assume only one mask
            assert len(masked_tokens) == 1
            
            # Selected only the masked tokens from the output
            vals = output.logits.cpu().numpy()
            predicted_masks = [vals[0][i] for i in masked_tokens]

            # Cache the mapping of the masked token only
            computed_embed_batches_test.append(predicted_masks)
        
        return computed_embed_batches_test

def predict_type(types_embed_array: np.array, types_embed_labels: np.array, indexed_knn: AnnoyIndex, k: int):
    """
    Predict type of given type embedding vectors
    """

    pred_types_embed = []
    pred_types_score = []
    for i, embed_vec in enumerate(tqdm(types_embed_array, total=len(types_embed_array), desc="Finding KNNs & Prediction")):
        
        # Get the distances to the KNN
        idx, dist = indexed_knn.get_nns_by_vector(embed_vec[0], k, include_distances=True)
        
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