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

We use the data from [Dreaddit: A Reddit Dataset for Stress Analysis in Social Media](https://aclanthology.org/D19-6213/). This data consists of Reddit posts from several subreddits with binary labels denoting whether a given post contains evidence of stress.

The Dreaddit data is available at http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip and can be downloaded as follows:

```
wget -P data http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip
unzip data/dreaddit.zip -d data
rm data/dreaddit.zip
```

## RoBERTa Models

We finetuned MentalRoBERTa models to predict stress or context (subreddit) of Reddit posts on the Dreaddit dataset. These two models can be downloaded as a zip file from here: https://drive.google.com/file/d/1iuClcReIJlunTKhmAyThI5UMwjNV3uu4/view?usp=sharing

After downloading them, unzip them with `unzip models.zip`

## MCTS Explanations

### Extract Explanations with MCTS

The script [`run_mcts.py`](https://github.com/swansonk14/MCTS_Interpretability/blob/main/run_mcts.py) trains or loads stress and context (subreddit) prediction models and then performs Monte Carlo tree search (MCTS) to identify context-dependent and context-independent explanations of stress in the text.

For example, the following command trains Multilayer Perceptron models and uses them in MCTS to extract explanations.

```
python run_mcts.py \
    --train_path data/dreaddit-train.csv \
    --test_path data/dreaddit-test.csv \
    --model_name mlp \
    --save_path explanations/mlp.pkl
```

The `model_name` can be `bnb` for Bernoulli Naive Bayes, `mnb` for Multinomial Naive Bayes, `mlp` for Multilayer Perceptron, `svm` for Support Vector Machine, or `roberta` for RoBERTa.

All scikit-learn models are trained and evaluated by the script while the RoBERTa model is loaded from a pretrained checkpoint. RoBERTa models therefore require the extra arguments `--stress_model_dir models/stress` and `--context_model_dir models/context` to load the pretrained RoBERTa models.

MCTS generates explanations in the form of binary masks over the input text. These explanations, along with stress and context entropy scores, are saved in a pickle file at `save_path`.

### Analyze Extracted Explanations

After generating MCTS explanations, the [`analyze_mcts_explanations.py`](https://github.com/swansonk14/MCTS_Interpretability/blob/main/analyze_mcts_explanations.py) script can be used to analyze the stress and context entropy of the extracted explanations, as demonstrated below.

```
python analyze_mcts_explanations.py \
    --explanations_path explanations/mlp.pkl \
    --save_dir analysis/mlp
```

Plots illustrating the distribution of stress and context entropy scores are saved in the `save_dir`.
