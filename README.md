# Hints on the Data for Language Modeling Experiments

This repository contains the code and data for the experiments conducted in the paper "Hints on the Data for Language Modeling". The paper explores different techniques for improving language modeling by providing hints about the structure of the data.

## Paper Abstract

Language Models (LM) are becoming more and more useful for providing representations upon which to train Natural Language Processing applications. However,  there is now clear evidence that attention-based transformers require a critical amount of language data to produce good enough LMs. The question we have addressed in this paper is to what extent the critical amount of data varies for languages of different morphological typology, in particular those that have a rich inflectional morphology, and whether the tokenization method to preprocess the data can make a difference. These details can be important for low-resource languages that need to plan the production of datasets. We evaluated intrinsically and extrinsically the differences of five different languages with different pretraining dataset sizes and three different tokenization methods for each. The results confirm that the size of the vocabulary due to morphological characteristics is directly correlated with both the LM perplexity and the performance of two typical downstream tasks such as NER identification and POS Tagging. The experiments also provide new evidence that a canonical tokenizer can reduce perplexity by more than a half for a polysynthetic language like Quechua as well as raising macro-F1 score from 0.8 to more than 0.9 in both downstream tasks with a LM trained with only 6M tokens.

## Contents

The repository contains the following files and directories:

- `data/`: This directory contains the preprocessed data used in the experiments, including the training, validation, and test sets.

- `models/`: This directory contains the trained language models for each experiment, as well as the scripts used to train and evaluate them.

- `results/`: This directory contains the results of the experiments, including the perplexity scores and other evaluation metrics.

- `scripts/`: This directory contains the scripts used to preprocess the data and extract the hints from it.

## Reproducing the Experiments

To reproduce the experiments from the paper, follow these steps:

1. Clone the repository:

```
git clone https://github.com/IULATERM-TRL-UPF/Hints-on-the-data-for-language-modeling/edit/main/README.md
```

2. Install the required dependencies:


Create and activate a virtualenv environment 
```
python -m venv venv
source venv/bin/activate
```

To install the required dependencies, run the following command in your terminal:

```
pip install -r requirements.txt
```

3. Train the language models:


Note that `model_name` and `results_name` should be replaced with the name of the model and the results file, respectively.

```
./train_model.sh <path_train> <path_eval> <language>
```

4. Fine-Tuned POS-Tagging:

```
./train_pos_tagging.sh <path_train> <path_eval> <checkpoint_model> <path_output>
```

## Contributing

Feel free to contribute to this repository by creating pull requests. If you find any bugs or have suggestions for improvements, please open an issue.

## License

This repository is licensed under the MIT License. See the LICENSE file for details.
