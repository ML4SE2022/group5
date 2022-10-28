# Group5

### Prepare the dataset

Due to size limitations, we do not include the dataset in this repository. To obtain the dataset and preprocess it:

First, enter the `dataset` directory:

```
cd dataset
```

Second, pull the Zenodo record of the dataset:

```
wget https://zenodo.org/record/6387001/files/ManyTypes4TypeScript.tar.gz?download=1 -O ManyTypes4TypeScript.tar.gz
```

Unzip the downloaded archive:

```
tar -xvzf ManyTypes4TypeScript.tar.gz
```

Finally, install the required dependencies and preprocess the data into the appropriate format using the `preprocess_dataset` python script:

```
pip install requirements.txt
python preprocess_dataset.py -v <vocab-size> 
```

Substitute `<vocab-size>` by the desired type vocabulary size. For reproduction purposes, use `50000`. Alternatively, the `ManyTypes4TypeScript` dataset can be used directly through python's `datasets` library. You can toggle this by using the `--use_local_dataset True` command line argument in the Docker container below.

### Run Docker

```
docker build -t typespacebert .
```

```
docker run --gpus all typespacebert [arguments]
```

In case GPUs are not recognized by the docker container, make sure `nvidia-container-toolkit` is installed and the docker daemon is restarted. For a better understanding of the available arguments, consolut the description below. For convenience, we provide several use cases that may be of interest:

1. Train `TypeSpaceBERT` from scratch on the full data set, using the same parameters as in the paper: `docker run --gpus all typespacebert --do_train True --custom_model_d 8 --use_full_dataset True`

2. Train our classification baseline from scratch on the full data set, using the same parameters as in the paper: `docker run --gpus all typespacebert --do_train True --use_classifier True --window_size 8 --use_full_dataset True`

3. Evaluate our provided `TypeSpaceBERT` on the full test set using the same parameters as in the paper: `docker run --gpus all typespacebert --do_eval True --window_size 8 --use_full_dataset True`

4. Evaluate our provided basesline model on the full test set using the same parameters as in the paper: `docker run --gpus all typespacebert --do_eval True --window_size 8 --use_full_dataset True`

5. If for any of the above commands, you would like to use our provided subset of data instead of the entirety of of the `ManyTypes4TypeScript` dataset, `--use_full_dataset` should simply be set to false.

#### Expected results

We expect that the results after 41 checkpoints (41000 training iterations) on the full data set, and 191 checkpoints (191000 iterations) of building the type space on the train set

### Run Manually

#### Dependency installation
```
pip install -r requirements.txt
```

### File structure and contents

```
📦group5
 ┣ 📂50k_types # Small dataset sample
 ┃ ┣ 📜test1.jsonl # Test set data
 ┃ ┣ 📜train4.jsonl # Train set data
 ┃ ┣ 📜valid1.jsonl # Validation set data
 ┃ ┗ 📜vocab_50000.txt # Type vocabulary for the entire data set
 ┣ 📂src
 ┃ ┣ 📜ManyTypes4TypeScript.py # Dataset utilities
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜train.py # Training and testing functionality
 ┃ ┣ 📜trainFunctions.py # Model functionality tools
 ┃ ┗ 📜typeSpace.py # Type Space generation tools
 ┣ 📜CodeBertExtension.ipynb # Interactive notebook
 ┣ 📜Dockerfile # Replication container file
 ┣ 📜README.md
 ┣ 📜__init__.py
 ┣ 📜requirements.txt
 ┗ 📜type4py_discussion_notes.md
```

### Dataset

This project uses the `ManyTypes4TypeScript` dataset. For further details, we recommend visitng the [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/TypePrediction-TypeScript) repository, the [Zenodo record](https://zenodo.org/record/6387001), or the consulting authors' [MSR '22 publication](https://www.kevinrjesse.com/pdfs/ManyTypes4TypeScript.pdf).


We provide a small subset of the data as a means of locally veryfing the functionality of our tool. This is the default option when running the container. To toggle to the full data set, consult the `arguments` section of this document.

### Fine-tuned models

The model trained to the specification described in our paper, together with its corresponding type space, and the fine-tuned classification baseline are all available on [Google Drive](https://drive.google.com/drive/folders/1-9SD27j9PFIpHO71G4Zduc-1CgCXuVqR).

### Arguments

```
--do_train (Bool). Defaults to False. Whether to train the model.
--do_eval (Bool). Defaults to False. Whether to run evaluation on the test set.
--do_valid (Bool). Defaults to False. Whether to run the evaluation on the validaiton set.
--use_classifier (Bool). Defaults to False. Whether to use the classification-based baseline model. When set to false, the DSL model will be considered instead.

--output_dir (string). Defaults to "type-model". The output directory where the model predictions and checkpoints will be written.

--train_batch_size (int). Defaults to 1000. Determines the batch size per GPU/CPU for training.
--eval_batch_size (int). Defaults to 4. Determines the batch size per GPU/CPU for evaluation.
--gradient_accumulation_steps (int). Defaults to 1. Determines the number of updates steps to accumulate before performing a backward/update pass.
--learning_rat (float). Defaults to 0.001. Specifices the initial learning rate for the Adam optimizer.
--weight_decay (float). Defaults to 0.0. Specifies the weight decay parameter, if any.
--adam_epsilon (float). Defaults to 1e-8. Determines the value of epsilon used for the Adam optimizer.
--max_grad_norm (float). Defaults to 1.0. Determines the max gradient norm used for training.
--num_train_epochs (int). Defaults to 1. Specifies the number of training epochs to perform.
--max_steps (int). Defaults to -1. If positive, specifies the total number of training steps to perform. It overrides --num_train_epochs.
--warmup_steps (int). Defaults to 0. Specifies the total number of warmup steps to perform linear warmup on.

--custom_model_d (int). Defaults to 8. Determines the output dimension of the DSL model, if it is used.

--interval (int). Defaults to 1000. Interval for outputting models
--knn_search_size (int). Defaults to 10. KNN seach size
--window_size (int). Defaults to 128. Window size used for tokenization              
--last_model (string). Defaults to "/model_intermediary40.pth". The TypeSpaceBERT model checkpoint to be used for evaluation
--last_class_model (string). Defaults to "/model_intermediary_classification9.pth". The TypeSpaceBERT model checkpoint to be used for evaluation
--local_dataset (bool). Defaults to True. True, if you want to run with the local dataset 
```