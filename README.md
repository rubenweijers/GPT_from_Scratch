This repository contains a basic implementation of a bigram language model using PyTorch. The model is trained on the text data provided in input.txt to predict the next character in a sequence.

**Key Features**
* Transformer Architecture: The model utilizes the Transformer architecture, known for its effectiveness in natural language processing tasks.
* PyTorch Framework: Implemented using PyTorch, a popular deep learning library for flexibility and ease of use.
* Hyperparameter Tuning: Several hyperparameters are exposed in the code for easy experimentation and customization.

**Usage**

1. Clone the repository:

Bash
```
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
   
Bash
```
pip install torch
```

3. Prepare data:

* Place your text data in a file named input.txt within the repository directory.
* The code assumes UTF-8 encoding for the input text.

4. Run the model:

Bash
```
python model.py
```
* The model will train on the text data and periodically print training and validation losses.
* After training, the model will generate a sample text based on the learned patterns.

**Code Structure**
```model.py```:  Contains the core PyTorch code for:

* Data preparation (encoding/decoding text, splitting data into training and validation sets).
* Model architecture (embedding layers, Transformer blocks, linear head).
* Training loop (loss calculation, optimization).
* Text generation.
```input.txt```: Dataset containing small Shakespeare dataset

**Customization**
You can adjust the model's behavior by modifying the following hyperparameters in the code:

* ```batch_size```: Number of sequences processed in parallel during training.
* ```block_size```: Maximum context length for predictions.
* ```max_iters```: Total number of training iterations.
* ```eval_interval```: Interval at which to evaluate and print losses.
* ```learning_rate```: Learning rate for the optimizer.
* ```n_embd```: Embedding dimension.
* ```n_head```: Number of attention heads in each Transformer block.
* ```n_layer```: Number of Transformer blocks.
* ```dropout```: Dropout probability.
