"""Runs MCTS to extract context-dependent and context-independent explanations from text."""
import pickle
from functools import partial
from pathlib import Path
from typing import Literal, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tqdm import tqdm

from mcts import get_best_mcts_node, get_phrases, MCTSExplainer


MODEL_NAMES = Literal['bnb', 'mnb', 'mlp', 'svm']
MODEL_TYPES = BernoulliNB | MultinomialNB | MLPClassifier | SVC


def train_sklearn_model(model_name: MODEL_NAMES,
                        train_text_counts: np.ndarray,
                        train_labels: np.ndarray) -> MODEL_TYPES:
    """Trains a scikit-learn model on the provided train token counts and labels.

    :param model_name: The name of the model to train.
    :param train_text_counts: A 2D array containing token counts for each training example (num_examples, num_tokens).
    :param train_labels: A 1D array of labels for each training example.
    :return: The trained model.
    """
    # Build model
    if model_name == 'bnb':
        model = BernoulliNB()
    elif model_name == 'mnb':
        model = MultinomialNB()
    elif model_name == 'mlp':
        model = MLPClassifier(random_state=0, max_iter=1000)
    elif model_name == 'svm':
        model = SVC(random_state=0, probability=True)
    else:
        raise ValueError(f'Model name "{model_name}" is not supported.')

    # Train model
    model.fit(train_text_counts, train_labels)

    return model


def evaluate_sklearn_model(model: MODEL_TYPES,
                           test_text_counts: np.ndarray,
                           test_labels: np.ndarray,
                           average: str) -> None:
    """Evaluates a scikit-learn model on the provided test token counts and labels.

    :param model: A trained scikit-learn model.
    :param test_text_counts: A 2D array containing token counts for each training example (num_examples, num_tokens).
    :param test_labels: A 1D array of labels for each training example.
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


"""
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Load RoBERTa
device = torch.device('cpu')
torch.set_grad_enabled(False)
softmax = torch.nn.Softmax(dim=-1)

tokenizer_roberta = RobertaTokenizer.from_pretrained('mental/mental-roberta-base', do_lower_case=True)

stress_model_roberta = RobertaForSequenceClassification.from_pretrained('/Users/kyleswanson/Stanford/three-irish-wannabe/models/stress', num_labels=2).to(device)
stress_model_roberta.eval()

domain_model_roberta = RobertaForSequenceClassification.from_pretrained('/Users/kyleswanson/Stanford/three-irish-wannabe/models/domain', num_labels=3).to(device)
domain_model_roberta.eval()


def stress_scoring_fn_roberta(text: str) -> float:
    tokens = tokenizer_roberta.encode(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    stress_logits = stress_model_roberta(tokens).logits
    stress_probs = softmax(stress_logits)
    stress_score = stress_probs[0, 1].item()

    return stress_score


def domain_entropy_scoring_fn_roberta(text: str) -> float:
    tokens = tokenizer_roberta.encode(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    domain_logits = domain_model_roberta(tokens).logits
    domain_probs = softmax(domain_logits)
    domain_entropy = entropy(domain_probs[0].detach().cpu().numpy())

    return domain_entropy
"""


def run_mcts(train_path: Path,
             test_path: Path,
             model_name: Literal['bnb', 'mnb', 'mlp', 'svm'],
             save_path: Path,
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

    print(f'Train size = {len(train_texts):,}')
    print(f'Num stressed = {np.sum(train_stress):,}\n')

    # Load test data
    test_data = pd.read_csv(test_path)
    test_texts, test_stress, test_subreddits = test_data['text'], test_data['label'], test_data['subreddit']

    print(f'Test size = {len(test_texts):,}')
    print(f'Num stressed = {np.sum(test_stress):,}\n')

    # Fit CountVectorizer on the train texts for stress model
    stress_count_vectorizer = CountVectorizer(ngram_range=ngram_range).fit(train_texts)

    # Convert train and test texts to counts
    train_text_counts = stress_count_vectorizer.transform(train_texts)
    test_text_counts = stress_count_vectorizer.transform(test_texts)

    # Train stress model
    print(f'Stress {model_name.upper()} model')
    stress_model = train_sklearn_model(
        model_name=model_name,
        train_text_counts=train_text_counts,
        train_labels=train_stress
    )

    # Evaluate stress model
    evaluate_sklearn_model(
        model=stress_model,
        test_text_counts=test_text_counts,
        test_labels=test_stress,
        average='binary'
    )

    # Filter to certain subreddits
    selected_subreddits = {'anxiety', 'relationships', 'assistance'}
    print(f'Filtering data to only contain these subreddits: {selected_subreddits}\n')

    # Filter training data to certain subreddits
    train_subreddit_mask = np.array([subreddit in selected_subreddits for subreddit in train_subreddits])
    train_texts_selected = train_texts[train_subreddit_mask]
    train_subreddits_selected = train_subreddits[train_subreddit_mask]
    print(f'Train size after filtering = {len(train_texts_selected):,}\n')

    # Filter test data to certain subreddits
    test_subreddit_mask = np.array([subreddit in selected_subreddits for subreddit in test_subreddits])
    test_texts_selected = test_texts[test_subreddit_mask]
    test_subreddits_selected = test_subreddits[test_subreddit_mask]
    print(f'Test size after filtering = {len(test_texts_selected):,}\n')

    # Fit CountVectorizer on selected train texts for context model
    context_count_vectorizer = CountVectorizer(ngram_range=ngram_range).fit(train_texts_selected)

    # Convert train and test texts to counts
    train_text_selected_counts = context_count_vectorizer.transform(train_texts_selected)
    test_text_selected_counts = context_count_vectorizer.transform(test_texts_selected)

    # Train subreddit model
    print(f'Context (subreddit) {model_name.upper()} model')
    context_model = train_sklearn_model(
        model_name=model_name,
        train_text_counts=train_text_selected_counts,
        train_labels=train_subreddits_selected
    )

    # Evaluate subreddit model
    evaluate_sklearn_model(
        model=context_model,
        test_text_counts=test_text_selected_counts,
        test_labels=test_subreddits_selected,
        average='macro'
    )

    # Define stress scoring function
    def stress_scoring_fn(text: str) -> float:
        counts = stress_count_vectorizer.transform([text])

        # Handle edge case of no recognized tokens
        if counts.count_nonzero() == 0:
            return 0.5

        stress_preds = stress_model.predict_proba(counts)
        stress_score = stress_preds[0, 1]

        return stress_score

    # Define context entropy scoring function
    def context_entropy_scoring_fn(text: str) -> float:
        counts = context_count_vectorizer.transform([text])

        # Handle edge case of no recognized tokens
        if counts.count_nonzero() == 0:
            return 0.5

        context_preds = context_model.predict_proba(counts)
        context_entropy = entropy(context_preds[0])

        return context_entropy

    # Define stress and context scoring function
    def stress_and_context_scoring_fn(text: str, context_dependent: bool) -> float:
        stress_score = stress_scoring_fn(text)
        context_entropy = context_entropy_scoring_fn(text)

        return stress_score + (-1 if context_dependent else 1) * alpha * context_entropy

    # Create an MCTSExplainer with the defined scoring function
    mcts_explainer = MCTSExplainer(
        max_phrases=max_phrases,
        min_phrase_length=min_phrase_length,
        scoring_fn=stress_and_context_scoring_fn,
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
        train_path: Path  # The path to the CSV file containing the train text and labels.
        test_path: Path  # The path to the CSV file containing the test text and labels.
        model_name: MODEL_NAMES  # The name of the model to train.
        save_path: Path  # The path to a pickle file where the explanations will be saved.
        alpha: float = 10.0  # The value of the parameter that weighs context entropy compared to stress.
        ngram_range: tuple[int, int] = (1, 1)  # The range of n-gram sizes for extracting token count features for scikit-learn models.
        max_phrases: int = 3  # Maximum number of phrases in an explanation.
        min_phrase_length: int = 5  # Minimum number of words in a phrase.
        n_rollout: int = 20  # The number of times to build the Monte Carlo tree.
        min_percent_unmasked: float = 0.2  # The minimum percent of words unmasked, used as a stopping point for leaf nodes in the search tree.
        max_percent_unmasked: float = 0.5  # The maximum percent of words that are unmasked.
        c_puct: float = 10.0  # The hyperparameter that encourages exploration.
        num_expand_nodes: int = 10  # The number of MCTS nodes to expand when extending the child nodes in the search tree.

    run_mcts(**Args().parse_args().as_dict())
