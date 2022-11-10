# TypeSpaceBERT
## Group5 - [paper](https://github.com/ML4SE2022/group5/blob/main/TypeSpaceBERT%20A%20Deep%20Similarity%20Learning-based%20CodeBERT%25aModel%20for%20Type%20Inference.pdf)

### Prepare the dataset

This project uses the `ManyTypes4TypeScript` dataset. For further details, we recommend visitng the [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/TypePrediction-TypeScript) repository, the [Zenodo record](https://zenodo.org/record/6387001), or the consulting authors' [MSR '22 publication](https://www.kevinrjesse.com/pdfs/ManyTypes4TypeScript.pdf). Due to size limitations, we do not include the dataset in this repository. To obtain the dataset and preprocess it:

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
pip install -r requirements.txt
python preprocess_dataset.py -v <vocab-size> 
```

Substitute `<vocab-size>` by the desired type vocabulary size. For reproduction purposes, use `50000`. Alternatively, the `ManyTypes4TypeScript` dataset can be used directly through python's `datasets` library. You can toggle this by using the `--use_local_dataset True` command line argument in the Docker container below.

### Fine-tuned models

The model trained to the specification described in our paper, together with its corresponding type space, and the fine-tuned classification baseline are all available on [Google Drive](https://drive.google.com/drive/folders/1-9SD27j9PFIpHO71G4Zduc-1CgCXuVqR).

To verify our fine-tuned models, download the corresponding model (either DSL or baseline) and its corresponding type space (if using the DSL approach) and place them in the `/models` directory. To specify a model to validate, use the `--use_model <model_name>` and `--use_typespace <typespace name>` command line arguments.

### Run Docker

```
docker build -t typespacebert .
```

```
docker run -v ${PWD}/models:/models --gpus all typespacebert [arguments]
```

In case GPUs are not recognized by the docker container, make sure `nvidia-container-toolkit` is installed and the docker daemon is restarted. For a better understanding of the available arguments, consolut the description below. For convenience, we provide several use cases that may be of interest (in terms of the arguments to provide the abvoce command with):

1. Train `TypeSpaceBERT` from scratch on the full data set, using the same parameters as in the paper:

   ```--do_train True --custom_model_d 8 --local_dataset True```

2. Train our classification baseline from scratch on the full data set, using the same parameters as in the paper:

   ```--do_train True --use_classifier True --window_size 8 --local_dataset True --custom_model_d 50000```

3. Evaluate our provided `TypeSpaceBERT` on the full test set using the same parameters as in the paper:

   ```--do_eval True --window_size 128 --local_dataset True --use_model typespacebert-model.pth --use_typespace typespacebert-type_space.ann```

4. Evaluate our provided basesline model on the full test set using the same parameters as in the paper:

   ```--do_eval True --window_size 8 --local_dataset True --use_model baseline-model.pth```

If for any of the above commands, you would like to use the remote version `ManyTypes4TypeScript` instead of preprocessing the data locally, `--local_dataset` should simply be set to `False`.

In some instances, we observed errors stemming from the `model = RobertaModel.from_pretrained("microsoft/codebert-base")` line in `train.py`. We were unable to identify the root cause of this, however, in all instance, re-building the container without caches before re-running our intended command (i.e., `docker build --no-cache -t typespacebert . && docker run -v ${PWD}/models:/models --gpus all typespacebert [arguments]`) solved the problem.

#### Expected results

We expect that the results after 41 checkpoints (41000 training iterations) on the full data set, and 191 checkpoints (191000 iterations) of building the type space on the train set to resemble the results we presented in Table 5.1 of our paper.

### Predictions

To predict single instance types you need to format the input data according to the following format:

   ```
      $ docker run -v ${PWD}/models:/models --rm --gpus all typespacebert --do_predict '["import", "{", "reactive", ",", "ref", ",", "watch", ",", "Ref", "}", "from", "'@vue/composition-api'", ";", "interface", "Options", "<", "T", ">", "{", "pendingDelay", "?", ":", "number", "|", "Ref", "<", "number", ">", ";", "promise", "?", ":", "Promise", "<", "T", ">", "|", "Ref", "<", "Promise", "<", "T", ">", ">", "|", "Ref", "<", "Promise", "<", "T", ">", "|", "null", ">", "|", "null", ";", "}", "export", "function", "usePromise", "<", "T", ">", "(", "options", "=", "{", "}", ")", "{", "const", "state", "=", "reactive", "(", "{", "promise", ":", "ref", "<", "Promise", "<", "T", ">", "|", "null", ">", "(", "options", ".", "promise", "||", "null", ")", ",", "isPending", ":", "ref", "(", "true", ")", ",", "data", ":", "ref", "<", "T", "|", "null", ">", "(", "null", ")", ",", "error", ":", "ref", "<", "Error", "|", "null", ">", "(", "null", ")", ",", "isDelayOver", ":", "ref", "(", "false", ")", ",", "}", ")", ";", "let", "timerId", "=", "null", ";", "const", "localOptions", "=", "reactive", "(", "{", "pendingDelay", ":", "options", ".", "pendingDelay", "==", "null", "?", "200", ":", "options", ".", "pendingDelay", ",", "}", ")", ";", "function", "setupDelay", "(", ")", "{", "if", "(", "localOptions", ".", "pendingDelay", ">", "0", ")", "{", "state", ".", "isDelayOver", "=", "false", ";", "if", "(", "timerId", ")", "clearTimeout", "(", "timerId", ")", ";", "timerId", "=", "setTimeout", "(", "(", ")", "=>", "(", "state", ".", "isDelayOver", "=", "true", ")", ",", "localOptions", ".", "pendingDelay", ")", ";", "}", "else", "{", "state", ".", "isDelayOver", "=", "true", ";", "}", "}", "watch", "(", "(", ")", "=>", "state", ".", "promise", ",", "newPromise", "=>", "{", "state", ".", "isPending", "=", "true", ";", "state", ".", "error", "=", "null", ";", "if", "(", "!", "newPromise", ")", "{", "state", ".", "data", "=", "null", ";", "state", ".", "isDelayOver", "=", "false", ";", "if", "(", "timerId", ")", "clearTimeout", "(", "timerId", ")", ";", "timerId", "=", "null", ";", "return", ";", "}", "setupDelay", "(", ")", ";", "newPromise", ".", "then", "(", "value", "=>", "{", "if", "(", "state", ".", "promise", "===", "newPromise", ")", "{", "state", ".", "data", "=", "value", ";", "state", ".", "isPending", "=", "false", ";", "}", "}", ")", ".", "catch", "(", "err", "=>", "{", "if", "(", "state", ".", "promise", "===", "newPromise", ")", "{", "state", ".", "error", "=", "err", ";", "state", ".", "isPending", "=", "false", ";", "}", "}", ")", ";", "}", ")", ";", "return", "{", "state", ",", "options", ":", "localOptions", ",", "set", ":", "(", "p", ")", "=>", "(", "state", ".", "promise", "=", "p", ")", ",", "}", ";", "}"]' '[null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, "Readonly", null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, "<MASK>", null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, "Promise", null, null, null, null, null, null, null, null, null, null, null, null, null]'
   ```

   #### Requirements
   - `--do_predict` signals that a prediction needs to be made based on the provided model and type space. It expects two arguments:
     - `input_ids` string of list of strings of code tokens.
     - `m_labels` string of list of strings with the correspondings labels for the code tokens. 
       - Insert `<MASK>` to mask a type for the model to predict
       - `null` can also be types, therefore make sure that `null` types have the corresponding quotations like `"null"` to be recognized.
     - A model called `typespacebert-model.pth` or specified using the arguments.
     - A type space called `typespacebert-type_space.ann` or specified using the arguments.
     - Make sure you have an input example that is larger then the used window size, otherwise no results will be returned.

   #### Result
   The result is printed out in the terminal in the following format, which indicates the predicted types with its corresponding confidence score: 

   ```bash
      PREDICTION: {'boolean': 0.15843932854290083, 'Props': 0.11041888897857288, 'string': 0.14256076847743712, 'number': 0.09823127675068363}
   ```

---

### Run Manually

#### Dependency installation
```
pip install -r requirements.txt
```

### File structure and contents

```
📦group5
 ┣ 📜__init__.py
 ┣ 📂dataset # Dataset data containing development size datasets (or the full downloaded dataset)
 ┃ ┣ 📜test1.jsonl # Test set data
 ┃ ┣ 📜train4.jsonl # Train set data
 ┃ ┣ 📜valid1.jsonl # Validation set data
 ┃ ┗ 📜vocab_50000.txt # Type vocabulary for the entire data set
 ┣ 📂src
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜ManyTypes4TypeScript.py # Dataset utilities
 ┃ ┣ 📜train.py # Training and testing functionality
 ┃ ┣ 📜trainFunctions.py # Model functionality tools
 ┃ ┗ 📜typeSpace.py # Type Space generation tools
 ┣ 📂models # Models and type spaces are stored here
 ┣ 📜CodeBertExtension.ipynb # Interactive notebook
 ┣ 📜Dockerfile # Replication container file
 ┣ 📜README.md
 ┣ 📜requirements.txt
```


We provide a small subset of the data as a means of locally veryfing the functionality of our tool. This is the default option when running the container. To toggle to the full data set, consult the `arguments` section of this document.



### Arguments

```
--do_train (bool). Defaults to False. Whether to train the model.
--do_eval (bool). Defaults to False. Whether to run evaluation on the test set.
--do_valid (bool). Defaults to False. Whether to run the evaluation on the validaiton set.
--use_classifier (bool). Defaults to False. Whether to use the classification-based baseline model. When set to false, the DSL model will be considered instead.

--use_model (string). Defaults to "". Indicates which model (specifically in the /models directory) to use. This must be specified when using validation.
--use_typespace (string). Defaults to "". Indicates which typespace (specifically in the /models directory) to use. This can be specified when using validation. If no typespace is given, one will be built from scratch, which may take very long.

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
--local_dataset (bool). Defaults to True. True, if you want to run with the local dataset 
```
