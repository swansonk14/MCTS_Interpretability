"""Runs MCTS to extract context-dependent and context-independent explanations from text."""
import pickle
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from mcts import get_best_mcts_node, get_phrases, MCTSExplainer


SKLEARN_MODELS = {'bnb', 'mnb', 'mlp', 'svm'}
SKLEARN_MODEL_NAMES = Literal['bnb', 'mnb', 'mlp', 'svm']
MODEL_NAMES = Literal['bnb', 'mnb', 'mlp', 'svm', 'roberta']
SKLEARN_MODEL_TYPES = BernoulliNB | MultinomialNB | MLPClassifier | SVC


def train_sklearn_model(model_name: SKLEARN_MODEL_NAMES,
                        train_text_counts: np.ndarray,
                        train_labels: list[int] | list[str]) -> SKLEARN_MODEL_TYPES:
    """Trains a scikit-learn model on the provided train token counts and labels.

    :param model_name: The name of the model to train.
    :param train_text_counts: A 2D array containing token counts for each training example (num_examples, num_tokens).
    :param train_labels: A list of labels for the training examples.
    :return: The trained model.
    """
    # Build model
    match model_name:
        case 'bnb':
            model = BernoulliNB()
        case 'mnb':
            model = MultinomialNB()
        case 'mlp':
            model = MLPClassifier(random_state=0, max_iter=1000)
        case 'svm':
            model = SVC(random_state=0, probability=True)
        case _:
            raise ValueError(f'Model name "{model_name}" is not supported.')

    # Train model
    model.fit(train_text_counts, train_labels)

    return model


def evaluate_sklearn_model(model: SKLEARN_MODEL_TYPES,
                           test_text_counts: np.ndarray,
                           test_labels: list[int] | list[str],
                           average: str) -> None:
    """Evaluates a scikit-learn model on the provided test token counts and labels.

    :param model: A trained scikit-learn model.
    :param test_text_counts: A 2D array containing token counts for each training example (num_examples, num_tokens).
    :param test_labels: A list of labels for the training examples.
    :param average: The type of averaging to perform to compute the metrics.
    """
    # Predict on the test set
    test_preds = model.predict(test_text_counts)

    # Evaluate predictions
    precision, recall, f1, support = precision_recall_fscore_support(test_labels, test_preds, average=average)
    accuracy = accuracy_score(test_labels, test_preds)

    # Print scores
    print(f'Precision = {precision:.3f}')
    print(f'Recall = {recall:.3f}')
    print(f'F1 = {f1:.3f}')
    print(f'Accuracy = {accuracy:.3f}\n')


def build_sklearn_model(model_name: SKLEARN_MODEL_NAMES,
                        train_texts: list[str],
                        train_labels: list[int] | list[str],
                        test_texts: list[str],
                        test_labels: list[int] | list[str],
                        ngram_range: tuple[int, int],
                        average: str) -> tuple[CountVectorizer, SKLEARN_MODEL_TYPES]:
    """Train and evaluate a scikit-learn model on text.

    :param model_name: The name of the model to train.
    :param train_texts: A list of train texts.
    :param train_labels: A list of train labels.
    :param test_texts: A list of test texts.
    :param test_labels: A list of test labels.
    :param ngram_range: The range of n-gram sizes for extracting token count features.
    :param average: The type of averaging to perform to compute the metrics.
    :return: A tuple containing a fitted CountVectorizer and a fitted scikit-learn model.
    """
    # Fit CountVectorizer on the train texts
    count_vectorizer = CountVectorizer(ngram_range=ngram_range)
    count_vectorizer.fit(train_texts)

    # Convert train and test texts to counts
    train_text_counts = count_vectorizer.transform(train_texts)
    test_text_counts = count_vectorizer.transform(test_texts)

    # Train stress model
    model = train_sklearn_model(
        model_name=model_name,
        train_text_counts=train_text_counts,
        train_labels=train_labels
    )

    # Evaluate stress model
    evaluate_sklearn_model(
        model=model,
        test_text_counts=test_text_counts,
        test_labels=test_labels,
        average=average
    )

    return count_vectorizer, model


def compute_score_sklearn(text: str,
                          count_vectorizer: CountVectorizer,
                          model: SKLEARN_MODEL_TYPES,
                          score_type: Literal['prediction', 'entropy']) -> float:
    """Computes the stress or context entropy score of text using a scikit-learn model applied to token counts.

    :param text: The text to evaluate.
    :param count_vectorizer: A CountVectorizer fit on the training texts that computes token counts.
    :param model: A trained scikit-learn model.
    :param score_type: The type of score to return.
                      'prediction' returns the prediction corresponding to the class at index 1
                      (i.e., the positive class for binary prediction).
                      'entropy' returns the entropy of the predictions across classes.
    :return: The score of the text according to the model.
    """
    counts = count_vectorizer.transform([text])

    # Handle edge case of no recognized tokens
    # Note: when score_type == 'entropy', a different value might be preferable
    if counts.count_nonzero() == 0:
        return 0.5

    preds = model.predict_proba(counts)

    match score_type:
        case 'prediction':
            score = preds[0, 1]
        case 'entropy':
            score = entropy(preds[0])
        case _:
            raise ValueError(f'Score type "{score_type}" is not supported.')

    return score


def compute_score_roberta(text: str,
                          tokenizer: RobertaTokenizer,
                          model: RobertaForSequenceClassification,
                          score_type: Literal['prediction', 'entropy']):
    """Computes the stress or context entropy score of text using a RoBERTa model.

    :param text: The text to evaluate.
    :param tokenizer: A RobertaTokenizer fit on the training texts.
    :param model: A trained RoBERTa model.
    :param score_type: The type of score to return.
                      'prediction' returns the prediction corresponding to the class at index 1
                      (i.e., the positive class for binary prediction).
                      'entropy' returns the entropy of the predictions across classes.
    :return: The score of the text according to the model.
    """
    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=512).to(model.device)
    logits = model(tokens).logits
    probs = F.softmax(logits, dim=-1)

    match score_type:
        case 'prediction':
            score = probs[0, 1].item()
        case 'entropy':
            score = entropy(probs[0].detach().cpu().numpy())
        case _:
            raise ValueError(f'Score type "{score_type}" is not supported.')

    return score


def compute_stress_and_context_entropy_score(text: str,
                                             context_dependent: bool,
                                             stress_scoring_fn: Callable[[str], float],
                                             context_entropy_scoring_fn: Callable[[str], float],
                                             alpha: float) -> float:
    """Computes the combined stress and context entropy score of a piece of text.

    :param text: The text to evaluate.
    :param context_dependent: If True, context entropy is negative,
                              so maximizing the score minimizes entropy (context-dependent).
                              If False, context entropy is positive,
                              so maximizing the score maximizes entropy (context-independent).
    :param stress_scoring_fn: A function that computes the stress score of a piece of text.
    :param context_entropy_scoring_fn: A function that computes the context entropy score of a piece of text.
    :param alpha: The value of the parameter that weighs context entropy compared to stress.
    :return: The sum of the stress and context entropy scores with the context entropy score
             multiplied by negative 1 if context_dependent is True and multiplied by the weight alpha.
    """
    stress_score = stress_scoring_fn(text)
    context_entropy = context_entropy_scoring_fn(text)

    return stress_score + (-1 if context_dependent else 1) * alpha * context_entropy


def run_mcts(train_path: Path,
             test_path: Path,
             model_name: MODEL_NAMES,
             save_path: Path,
             stress_model_dir: Optional[Path] = None,
             context_model_dir: Optional[Path] = None,
             device: Optional[int] = None,
             alpha: float = 10.0,
             ngram_range: tuple[int, int] = (1, 1),
             max_phrases: int = 3,
             min_phrase_length: int = 5,
             n_rollout: int = 20,
             min_percent_unmasked: float = 0.2,
             max_percent_unmasked: float = 0.5,
             c_puct: float = 10.0,
             num_expand_nodes: int = 10) -> None:
    """Runs MCTS to extract context-dependent and context-independent explanations from text.

    :param train_path: The path to the CSV file containing the train text and labels.
    :param test_path: The path to the CSV file containing the test text and labels.
    :param model_name: The name of the model to train.
    :param save_path: The path to a pickle file where the explanations will be saved.
    :param stress_model_dir: Path to directory containing a RoBERTa config.json and pytorch_model.bin
                             trained on stress (if model_name == 'roberta').
    :param context_model_dir: Path to directory containing a RoBERTa config.json and pytorch_model.bin
                              trained on context (if model_name == 'roberta').
    :param device: The GPU device on which to run the model. If None, defaults to CPU.
                   Only applicable if model_name == 'roberta'.
    :param alpha: The value of the parameter that weighs context entropy compared to stress.
    :param ngram_range: The range of n-gram sizes for extracting token count features for scikit-learn models.
    :param max_phrases: Maximum number of phrases in an explanation.
    :param min_phrase_length: Minimum number of words in a phrase.
    :param n_rollout: The number of times to build the Monte Carlo tree.
    :param min_percent_unmasked: The minimum percent of words unmasked,
                                 used as a stopping point for leaf nodes in the search tree.
    :param max_percent_unmasked: The maximum percent of words that are unmasked.
    :param c_puct: The hyperparameter that encourages exploration.
    :param num_expand_nodes: The number of MCTS nodes to expand when extending the child nodes in the search tree.
    """
    # Load train data
    train_data = pd.read_csv(train_path)
    train_texts, train_stress, train_subreddits = train_data['text'], train_data['label'], train_data['subreddit']
    num_train_stressed = np.sum(train_stress)

    print(f'Stress train size = {len(train_texts):,}')
    print(f'Num stressed = {num_train_stressed:,} ({100 * num_train_stressed / len(train_texts):.1f}%)\n')

    # Load test data
    test_data = pd.read_csv(test_path)
    test_texts, test_stress, test_subreddits = test_data['text'], test_data['label'], test_data['subreddit']
    num_test_stressed = np.sum(test_stress)

    print(f'Stress test size = {len(test_texts):,}')
    print(f'Num stressed = {num_test_stressed:,} ({100 * num_test_stressed / len(test_texts):.1f}%)\n')

    # Filter data to certain subreddits
    selected_subreddits = {'anxiety', 'relationships', 'assistance'}
    print(f'Filtering context data to only contain these subreddits: {selected_subreddits}\n')

    # Filter training data to certain subreddits
    train_subreddit_mask = np.array([subreddit in selected_subreddits for subreddit in train_subreddits])
    train_texts_selected = train_texts[train_subreddit_mask]
    train_subreddits_selected = train_subreddits[train_subreddit_mask]
    print(f'Context train size = {len(train_texts_selected):,}')
    for subreddit, count in train_subreddits_selected.value_counts().items():
        print(f'Num {subreddit} = {count:,} ({100 * count / len(train_texts_selected):.1f}%)')
    print()

    # Filter test data to certain subreddits
    test_subreddit_mask = np.array([subreddit in selected_subreddits for subreddit in test_subreddits])
    test_texts_selected = test_texts[test_subreddit_mask]
    test_subreddits_selected = test_subreddits[test_subreddit_mask]
    print(f'Context test size = {len(test_texts_selected):,}')
    for subreddit, count in test_subreddits_selected.value_counts().items():
        print(f'Num {subreddit} = {count:,} ({100 * count / len(test_texts_selected):.1f}%)')
    print()

    # Build models
    if model_name in SKLEARN_MODELS:
        # Ensure model directories are not provided accidentally
        if stress_model_dir is not None or context_model_dir is not None or device is not None:
            raise ValueError('Model directories and/or device should not be provided for scikit-learn models, '
                             'only for RoBERTa models.')

        print(f'Training and evaluating stress {model_name.upper()} model')

        # Train and evaluate stress model
        stress_count_vectorizer, stress_model = build_sklearn_model(
            model_name=model_name,
            train_texts=train_texts,
            train_labels=train_stress,
            test_texts=test_texts,
            test_labels=test_stress,
            ngram_range=ngram_range,
            average='binary'
        )

        # Define stress scoring function
        stress_scoring_fn = partial(
            compute_score_sklearn,
            count_vectorizer=stress_count_vectorizer,
            model=stress_model,
            score_type='prediction'
        )

        print(f'Training and evaluating context (subreddit) {model_name.upper()} model')

        # Train and evaluate context model
        context_count_vectorizer, context_model = build_sklearn_model(
            model_name=model_name,
            train_texts=train_texts_selected,
            train_labels=train_subreddits_selected,
            test_texts=test_texts_selected,
            test_labels=test_subreddits_selected,
            ngram_range=ngram_range,
            average='macro'
        )

        # Define context entropy scoring function
        context_entropy_scoring_fn = partial(
            compute_score_sklearn,
            count_vectorizer=context_count_vectorizer,
            model=context_model,
            score_type='entropy'
        )
    elif model_name == 'roberta':
        device = torch.device(device if device is not None else 'cpu')
        torch.set_grad_enabled(False)
        tokenizer = RobertaTokenizer.from_pretrained('mental/mental-roberta-base', do_lower_case=True)

        print('Loading stress RoBERTa model')

        # Load stress model
        stress_model = RobertaForSequenceClassification.from_pretrained(stress_model_dir, num_labels=2)
        stress_model.to(device).eval()

        # Define stress scoring function
        stress_scoring_fn = partial(
            compute_score_roberta,
            tokenizer=tokenizer,
            model=stress_model,
            score_type='prediction'
        )

        print('Loading context (subreddit) RoBERTa model')

        # Load context model
        context_model = RobertaForSequenceClassification.from_pretrained(context_model_dir, num_labels=3)
        context_model.to(device).eval()

        # Define context entropy scoring function
        context_entropy_scoring_fn = partial(
            compute_score_roberta,
            tokenizer=tokenizer,
            model=context_model,
            score_type='entropy'
        )
    else:
        raise ValueError(f'Model name "{model_name} is not supported.')

    # Define stress and context entropy scoring function
    stress_and_context_entropy_scoring_fn = partial(
        compute_stress_and_context_entropy_score,
        stress_scoring_fn=stress_scoring_fn,
        context_entropy_scoring_fn=context_entropy_scoring_fn,
        alpha=alpha
    )

    # Create an MCTSExplainer with the defined scoring function
    mcts_explainer = MCTSExplainer(
        max_phrases=max_phrases,
        min_phrase_length=min_phrase_length,
        scoring_fn=stress_and_context_entropy_scoring_fn,
        n_rollout=n_rollout,
        min_percent_unmasked=min_percent_unmasked,
        c_puct=c_puct,
        num_expand_nodes=num_expand_nodes
    )

    # Get all the test texts from the selected subreddits that are stressed
    test_stress_selected = test_stress[test_subreddit_mask]
    stressed_test_texts = test_texts_selected[test_stress_selected == 1]

    # Create lists to store the original, dependent, and independent stress and entropy for each explanation
    original_stress = []
    original_entropy = []

    masked_stress_dependent = []
    masked_entropy_dependent = []

    masked_stress_independent = []
    masked_entropy_independent = []

    # Run MCTS on each text in both context-dependent and context-independent manners
    for text in tqdm(stressed_test_texts):
        original_stress.append(stress_scoring_fn(text))
        original_entropy.append(context_entropy_scoring_fn(text))

        for context_dependent, masked_stress, masked_entropy in [
            (True, masked_stress_dependent, masked_entropy_dependent),
            (False, masked_stress_independent, masked_entropy_independent)
        ]:
            mcts_nodes = mcts_explainer.explain(text=text, context_dependent=context_dependent)
            best_mcts_node = get_best_mcts_node(mcts_nodes, max_percent_unmasked=max_percent_unmasked)
            words = best_mcts_node.words
            mask = best_mcts_node.mask

            text_phrases = [' '.join(words[i] for i in phrase) for phrase in get_phrases(mask)]

            masked_stress.append(np.mean([stress_scoring_fn(text_phrase) for text_phrase in text_phrases]))
            masked_entropy.append(np.mean([context_entropy_scoring_fn(text_phrase) for text_phrase in text_phrases]))

    # Save the explanations
    results = {
        'texts': stressed_test_texts,
        'original_stress': np.array(original_stress),
        'masked_stress_dependent': np.array(masked_stress_dependent),
        'masked_stress_independent': np.array(masked_stress_independent),
        'original_entropy': np.array(original_entropy),
        'masked_entropy_dependent': np.array(masked_entropy_dependent),
        'masked_entropy_independent': np.array(masked_entropy_independent)
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    from tap import Tap

    class Args(Tap):
        train_path: Path
        """The path to the CSV file containing the train text and labels."""
        test_path: Path
        """The path to the CSV file containing the test text and labels."""
        model_name: MODEL_NAMES
        """The name of the model to train."""
        save_path: Path
        """The path to a pickle file where the explanations will be saved."""
        stress_model_dir: Optional[Path] = None
        """Path to directory containing a RoBERTa config.json and pytorch_model.bin
           trained on stress (if model_name == 'roberta')."""
        context_model_dir: Optional[Path] = None
        """Path to directory containing a RoBERTa config.json and pytorch_model.bin
           trained on context (if model_name == 'roberta')."""
        device: Optional[int] = None
        """The GPU device on which to run the model. If None, defaults to CPU.
           Only applicable if model_name == 'roberta'."""
        alpha: float = 10.0
        """The value of the parameter that weighs context entropy compared to stress."""
        ngram_range: tuple[int, int] = (1, 1)
        """The range of n-gram sizes for extracting token count features for scikit-learn models."""
        max_phrases: int = 3
        """Maximum number of phrases in an explanation."""
        min_phrase_length: int = 5
        """Minimum number of words in a phrase."""
        n_rollout: int = 20
        """The number of times to build the Monte Carlo tree."""
        min_percent_unmasked: float = 0.2
        """The minimum percent of words unmasked, used as a stopping point for leaf nodes in the search tree."""
        max_percent_unmasked: float = 0.5
        """The maximum percent of words that are unmasked."""
        c_puct: float = 10.0
        """The hyperparameter that encourages exploration."""
        num_expand_nodes: int = 10
        """The number of MCTS nodes to expand when extending the child nodes in the search tree."""

    run_mcts(**Args().parse_args().as_dict())
