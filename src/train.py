from annoy import AnnoyIndex
import torch
import argparse
import os
import numpy as np
import torchmetrics

from datasets import load_dataset
from transformers import RobertaModel, RobertaTokenizerFast
from trainFunctions import CustomModel, TripletDataset, TripletLoss, classification_prediction, tokenize_and_align_labels
from tqdm import tqdm

from typeSpace import create_type_space, map_type, predict_type, DISTANCE_METRIC

# TODO: add instructions for pulling and integrating data set and model
MASKED_TOKEN_ID = 50264

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default="type-model", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--use_classifier", default=False, type=bool,
                        help="Whether to validate a classificaiton model.")
    parser.add_argument("--do_train", default=False, type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=False, type=bool,
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_valid", default=False, type=bool,
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--use_model", default="", type=str,
                        help="Provide a model to evaluate. The model is assumed to be in the /models directory.")
    parser.add_argument("--use_typespace", default="", type=str,
                    help="Provide a type space for evaluation. The type space is assumed to be in the /models directory.")

    parser.add_argument("--train_batch_size", default=1000, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=0.001, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--custom_model_d", default=8, type=int,
                        help="Out dimension of the custom model")
    parser.add_argument("--interval", default=1000, type=int,
                        help="Interval for outputting models")
    parser.add_argument("--knn_search_size", default=10, type=int,
                        help="KNN seach size")
    parser.add_argument("--window_size", default=128, type=int,
                        help="Window size used for tokenization")              
    parser.add_argument("--last_model", default="/model_intermediary40.pth", type=str,
                        help="The TypeSpaceBERT model checkpoint to be used for evaluation")
    parser.add_argument("--last_class_model", default="/model_intermediary_classification9.pth", type=str,
                        help="The TypeSpaceBERT model checkpoint to be used for evaluation")
    parser.add_argument("--local_dataset", default=True, type=bool,
                        help="True, if you want to run with the local dataset")   


    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train(args)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.local_dataset:
        dataset = load_dataset('ManyTypes4TypeScript.py', ignore_verifications=True)
    else:
        dataset = load_dataset('kevinjesse/ManyTypes4TypeScript')

    #load the small selected local dataset using the py script
    # TODO: fix the GitHub dataset
    
    model = RobertaModel.from_pretrained("microsoft/codebert-base")

    WINDOW = args.window_size
    tokenized_hf = dataset.map(tokenize_and_align_labels, batched=True, batch_size=args.train_batch_size, remove_columns=['id', 'tokens', 'labels'], fn_kwargs={"window": WINDOW})
      
    if args.do_train:  
        epochs = args.num_train_epochs

        custom_model = CustomModel(model, args.custom_model_d, codebert_output_dim = 768 * args.window_size, input_dim = args.window_size)
        labels = torch.tensor(tokenized_hf['train']['masks'])
        dataset = TripletDataset(torch.tensor(tokenized_hf['train']['input_ids']), m_labels=torch.tensor(tokenized_hf['train']['m_labels']), labels=labels, dataset_name="train")

        optimizer = torch.optim.Adam(custom_model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        criterion = torch.jit.script(TripletLoss()) if not args.use_classifier else torch.jit.script(torch.nn.CrossEntropyLoss())

        # TODO: re-add support for the classification model
        for epoch in tqdm(range(epochs), desc="Epochs"):
            custom_model.train()
            for step in tqdm(range(len(dataset)), desc="Steps"):
                # Skip instances that have only a signle occurance.
                mask = labels == labels[step]
                mask[step] = False # Making sure that the similar pair is NOT the same as the given index
                if len(mask.nonzero()) == 0:
                    continue
                INTERVAL = args.interval
                if step > 0 and step % INTERVAL == 0:
                    if args.output_dir is not None:
                        torch.save(custom_model, args.output_dir + "/model_intermediary" + ("_classification" if args.use_classifier else "") + str(count) + ".pth")
                    count += 1
                if not args.use_classifier: 
                    (t_a, t_p, t_n) = dataset.get_item_func(step)
                                
                    optimizer.zero_grad()
                    anchor_out = custom_model(input_ids=torch.cat((t_a[0][0], t_a[1]), 0))
                    positive_out = custom_model(input_ids=torch.cat((t_p[0][0], t_p[1]), 0))
                    negative_out = custom_model(input_ids=torch.cat((t_n[0][0], t_n[1]), 0))

                    loss = criterion(anchor_out, positive_out, negative_out)
                else:
                    t = dataset.get_single_item(step)
                    optimizer.zero_grad()
                    t_out = custom_model(input_ids=torch.cat((t[0][0], t[1]), 0))
                    # Get the position of the BERT-specific separation token.
                    masked_label_position = (t[1] == MASKED_TOKEN_ID).nonzero(as_tuple=True)[0].item()
                    target = torch.nn.functional.one_hot(torch.tensor([masked_label_position]), args.custom_model_d).squeeze(0).double()
                    
                    loss = criterion(t_out, target)             
                loss.backward()
                optimizer.step()
        if args.output_dir is not None:
            torch.save(custom_model, args.output_dir + f"/model{'_classification' if args.use_classifier else ''}.pth")
    
    if args.do_eval:
        assert args.use_model != "", "Please provide the model name to evaluate by using the --use_model command line argument."

        if not args.use_classifier:

            custom_model = torch.load("models/" + args.use_model) 
            custom_model.eval()
            if not os.path.isfile("models/" + args.use_typesapce):
                print("WARN: type space not provided. To provide a type space, use the --use_typespace command line argument.")
                space, computed_mapped_labels_train = create_type_space(custom_model, torch.tensor(tokenized_hf['train']['input_ids']), torch.tensor(tokenized_hf['train']['m_labels']), torch.tensor(tokenized_hf['train']['masks']))
                space.save(args.output_dir + '/space.ann')
            else:
                space = AnnoyIndex(8, DISTANCE_METRIC)
                space.load("models/" + args.use_model)
                computed_mapped_labels_train = []
                for label in torch.tensor(tokenized_hf['train']['masks']):
                    computed_mapped_labels_train.append(label)
        else:
            custom_model = torch.load("models/" + args.use_model) 
            custom_model.eval()

        custom_model = torch.load(args.output_dir + "/model_intermediary" + ("_classification" if args.use_classifier else "") + str(n) + ".pth")
        custom_model.eval()

        if not args.use_classifier:
            mapped_types_test = map_type(custom_model, torch.tensor(tokenized_hf['test']['input_ids'][:10000]), torch.tensor(tokenized_hf['test']['m_labels'][:10000]))
            KNN_SEARCH_SIZE = args.knn_search_size
            _, pred_types_score = predict_type(mapped_types_test, computed_mapped_labels_train, space, KNN_SEARCH_SIZE)

            
            with open("50k_types/vocab_50000.txt") as f:
                lines = dict(enumerate(f.readlines()))
                predictions = dict()
                for p in pred_types_score[0]:
                    predictions[p[0]] = p[1]

            preds = torch.tensor([ [ p[0] for p in prediction ] for prediction in pred_types_score ])
        else:
            preds = [classification_prediction(custom_model, torch.tensor(tokenized_hf['test']['input_ids'][i]), torch.tensor(tokenized_hf['test']['m_labels'][i])) for i in range(10000)]            # print(preds)
        target = torch.tensor(tokenized_hf['test']['masks'])
        
        print("EXACT MATCH ACCURACY:")
        
        true_pos = 0
        count = -1
        mrr = 0
        for prediction in preds:
            count += 1
            rank_i = 0
            for i, p in enumerate(prediction):
                if p == target[count]:
                    if rank_i == 0:
                        rank_i = i + 1
                    true_pos += 1
                    break
            if rank_i != 0:
                mrr += 1 / rank_i
        mrr = mrr / count
        accuracy_8 = true_pos / count
        
        
        true_pos = 0
        top_1 = []
        count = -1
        for prediction in preds:
            count += 1
            # top_1.append(prediction[0].item())
            if prediction[0] == target[count]:
                true_pos += 1
        accuracy = true_pos / count
    
        print((accuracy_8, mrr, accuracy))

if __name__ == "__main__":
    main()
