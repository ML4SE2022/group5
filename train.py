import torch
import argparse
import os

from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset
from transformers import RobertaModel
from trainFunctions import CustomModel, TripletDataset, TripletLoss, tokenize_and_align_labels
from tqdm.notebook import tqdm


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default="type-model", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--do_train", default=False, type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=False, type=bool,
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_valid", default=False, type=bool,
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--train_batch_size", default=4, type=int,
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

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train(args)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Uncomment if you want to download the full dataset from hugging face
    #dataset = load_dataset ( ' kevinjesse /ManyTypes4TypeScript ')

    #load the small selected local dataset using the py script 
    dataset = load_dataset('ManyTypes4TypeScript.py', ignore_verifications=True)
    
    print("Dataset loaded")

    model = RobertaModel.from_pretrained("microsoft/codebert-base")

    tokenized_hf = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=['id', 'tokens', 'labels'])

    print("Finished input tokenization")
    
    #TODO: begins
    epochs = args.num_train_epochs

    custom_model = CustomModel(model, args.custom_model_d)
    dataset = TripletDataset(torch.tensor(tokenized_hf['train']['input_ids']), m_labels=torch.tensor(tokenized_hf['train']['m_labels']), labels=torch.tensor(tokenized_hf['train']['masks']), dataset_name="train")

    optimizer = torch.optim.Adam(custom_model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    criterion = torch.jit.script(TripletLoss())

    for epoch in tqdm(range(epochs), desc="Epochs"):
        custom_model.train()
        running_loss = []
        for step in range(len(dataset)):
            (t_a, t_p, t_n) = dataset.get_item_func(step)
            
            optimizer.zero_grad()
            anchor_out = custom_model(input_ids=torch.cat((t_a[0][0], t_a[1]), 0))
            positive_out = custom_model(input_ids=torch.cat((t_p[0][0], t_p[1]), 0))
            negative_out = custom_model(input_ids=torch.cat((t_n[0][0], t_n[1]), 0))
            
            print(anchor_out)
            
            loss = criterion(anchor_out[0], positive_out[0], negative_out[0])
            loss.backward()
            optimizer.step()
    #TODO: Ends




if __name__ == "__main__":
    main()
