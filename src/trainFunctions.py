from torch.utils.data import TensorDataset
from typing import Tuple
import torch
from transformers import RobertaTokenizerFast
from datasets import load_dataset

#WINDOW = 128

def classification_prediction(model, inp, labels, k = 8):
    model_output = model(torch.cat((inp, labels), 0))
    probs = torch.nn.Softmax(model_output).dim
    _, indices = torch.topk(probs, k)

    return indices.tolist()
    
def divide_chunks(l1, l2, n):
    for i in range(0, len(l1), n):
        yield {'input_ids': [0] + l1[i:i + n] + [2], 'labels': [-100] + l2[i:i + n] + [-100]}

def tokenize_prediction(example, window):
    #fast tokenizer for roberta - please stick to the fast one or expect bugs and slowdown
    tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base", add_prefix_space=True)

    window_size = window - 2

    # TODO: fix tokenization parsing issue
    tokenized_inputs = tokenizer(example['tokens'], is_split_into_words=True, truncation=True)
    inputs = {'input_ids': [], 'm_labels': []}
    inputs_ = {'input_ids': [], 'm_labels': []}
    lines = dict()

    # Take first encoding since only one example is given.
    inputs['input_ids'] = tokenized_inputs.encodings[0].ids

    with open("50k_types/vocab_50000.txt") as f:
        lines = dict(enumerate(f.readlines()))

    keys = list(lines.keys())
    values = lines.values()
    values = list(map(lambda x: x.replace("\n", ""), values))

    for label in example['labels']:
        if label == '<MASK>':
            inputs['m_labels'].append(tokenizer.mask_token_id)
        if label == '<NULLTYPE>':
            inputs['m_labels'].append(-100)
        if label in values:
            inputs['m_labels'].append(keys[values.index(label)])
        else:
            inputs['m_labels'].append(-100)
    
    inputs['m_labels'] = inputs['m_labels']

    for e in divide_chunks(inputs['input_ids'], inputs['m_labels'], window_size):
        for k, v in e.items():
            if k == 'labels':
                k = 'm_labels'
            inputs_[k].append(v)

    inputs_new = {'input_ids': [], 'm_labels': []}
    for i in range(len(inputs_['input_ids'])):
        if len(inputs_['input_ids'][i]) != WINDOW or len(inputs_['m_labels'][i]) != WINDOW:
            continue
        inputs_new['input_ids'].append(inputs_['input_ids'][i])
        inputs_new['m_labels'].append(inputs_['m_labels'][i])

    return inputs_new

def tokenize_and_align_labels(examples, window):
    def divide_chunks(l1, l2, n):
        for i in range(0, len(l1), n):
            yield {'input_ids': [0] + l1[i:i + n] + [2], 'labels': [-100] + l2[i:i + n] + [-100]}

    #fast tokenizer for roberta - please stick to the fast one or expect bugs and slowdown
    tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base", add_prefix_space=True)

    window_size = window - 2
    # TODO: fix tokenization parsing issue
    tokenized_inputs = tokenizer(examples['tokens'], is_split_into_words=True, truncation=True)
    inputs_ = {'input_ids': [], 'labels': []}

    for encoding, label in zip(tokenized_inputs.encodings, examples['labels']):
        word_ids = encoding.word_ids  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                l = label[word_idx] if label[word_idx] is not None else -100
                label_ids.append(l)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        s_labels = set(label_ids)
        if len(s_labels) == 1 and list(s_labels)[0] == -100:
            continue
        for e in divide_chunks(encoding.ids, label_ids, window_size):
            for k, v in e.items():
                inputs_[k].append(v)

    inputs_new = {'input_ids': [], 'm_labels': [], "masks": []}

    for i in range(len(inputs_['labels'])):
        if len(inputs_['input_ids'][i]) != window:
            continue    
        for j in range(len(inputs_['labels'][i])):
            if inputs_['labels'][i][j]==-100:
                continue
            copy_label = inputs_['labels'][i].copy()
            copy_label[j] = tokenizer.mask_token_id
            inputs_new['input_ids'].append(inputs_['input_ids'][i])
            inputs_new['m_labels'].append(copy_label)
            inputs_new['masks'].append(inputs_['labels'][i][j])
    return inputs_new


class CustomModel(torch.nn.Module):
    def __init__(self, model, d, codebert_output_dim, input_dim): # 50265 + sep + 512 (labels) = 50778
        super(CustomModel, self).__init__() 
        self.d = d
        self.model = model
        self.config = model.config
        self.layer = torch.nn.Linear(codebert_output_dim + input_dim, d)
        self.input_dim = input_dim
        self.codebert_output_dim = codebert_output_dim
    
    def forward(self, input_ids=None, attention_mask=None):
        
        assert input_ids.shape[0] == self.input_dim * 2
        
        tokens, labels = torch.split(input_ids, self.input_dim)
        
        model_output = self.model.forward(input_ids=tokens.unsqueeze(0))[0]
        
        ll_input = torch.cat((model_output.view(1, self.codebert_output_dim).squeeze(0), labels), 0)
        assert ll_input.shape[0] == self.codebert_output_dim + self.input_dim
        
        final_output_tensor = self.layer.forward(ll_input)
        
        return final_output_tensor


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(0)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

class TripletDataset(torch.utils.data.Dataset):
    
    def __init__(self, *in_sequences: torch.Tensor, m_labels: torch.Tensor, labels: torch.Tensor, dataset_name: str,
                 train_mode: bool=True):
        self.data = TensorDataset(*in_sequences)
        self.m_labels = m_labels
        self.labels = labels
        self.dataset_name = dataset_name
        self.train_mode = train_mode

        self.get_item_func = self.get_item_train if self.train_mode else self.get_item_test

    def get_single_item(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.data[index], self.m_labels[index])

    def get_item_train(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                         Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        It returns three tuples. Each one is a (data, label)
         - The first tuple is (data, label) at the given index
         - The second tuple is similar (data, label) to the given index
         - The third tuple is different (data, label) from the given index 
        """

         # Find a similar datapoint randomly
        mask = self.labels == self.labels[index]
        mask[index] = False # Making sure that the similar pair is NOT the same as the given index
        mask = mask.nonzero()
        a = mask[torch.randint(high=len(mask), size=(1,))][0]

        # Find a different datapoint randomly
        mask = self.labels != self.labels[index]
        mask = mask.nonzero()
        b = mask[torch.randint(high=len(mask), size=(1,))][0]
        
        return (self.data[index], self.m_labels[index]), (self.data[a.item()], self.m_labels[a.item()]), \
               (self.data[b.item()], self.m_labels[b.item()])

    def get_item_test(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], list, list]:
        return (self.data[index], self.labels[index]), [], []
    
    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                         Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
         return self.get_item_func(index)

    def __len__(self) -> int:
        return len(self.data)
