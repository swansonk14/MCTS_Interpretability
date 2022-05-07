# Monte Carlo Tree Search for Interpreting Stress in Natural Language

This code accompanies the paper [Monte Carlo Tree Search for Interpreting Stress in Natural Language](https://arxiv.org/abs/2204.08105) by Kyle Swanson, Joy Hsu, and Mirac Suzgun, published at the [Second Workshop on Language Technology for Equality, Diversity, Inclusion (LT-EDI-2022)](https://sites.google.com/view/lt-edi-2022/home?authuser=0).

## Installation

Install conda environment.
```
conda env create -f environment.yml
```

Activate environment.
```
conda activate mcts_interpretability
```

## Data

We use the data from [Dreaddit: A Reddit Dataset for Stress Analysis in Social Media](https://aclanthology.org/D19-6213/). This data is available at http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip and can be downloaded as follows:

```
wget -P data http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip
unzip data/dreaddit.zip -d data
rm data/dreaddit.zip
```

## MCTS Explanations

Run MCTS on Dreaddit posts to extract context-dependent and context-independent explanations of stress. The `model_name` can be `bnb` for Bernoulli Naive Bayes, `mnb` for Multinomial Naive Bayes, `mlp` for Multilayer Perceptron, or `svm` for Support Vector Machine. Explanations (in the form of binary masks over the input text), along with stress and context entropy scores, are saved as pickle files in the `save_dir`.

```
python run_mcts.py \
    --train_path data/dreaddit-train.csv \
    --test_path data/dreaddit-test.csv \
    --model_name mlp \
    --save_path explanations/mlp/explanations.pkl
```

Analyze the MCTS explanations in terms of stress and context entropy.

```
python analyze_mcts_explanations.py \
    --explanations_path explanations/mlp/explanations.pkl \
    --save_dir explanations/mlp
```
