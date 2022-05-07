"""Runs MCTS to extract context-dependent and context-independent explanations from text."""
import pickle
from functools import partial
from pathlib import Path
from typing import Literal, Set, Tuple

import numpy as np
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


def load_data_and_embeddings(embeddings_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(embeddings_path, 'rb') as f:
        embedding_dicts = pickle.load(f)

    texts = np.array([embedding_dict['text'] for embedding_dict in embedding_dicts])
    embeddings_path = np.array([np.mean(embedding_dict['last_hidden_state'], axis=0) for embedding_dict in embedding_dicts])
    stress = np.array([embedding_dict['label'] for embedding_dict in embedding_dicts])
    subreddits = np.array([embedding_dict['subreddit'] for embedding_dict in embedding_dicts])

    return texts, embeddings_path, stress, subreddits


def train_sklearn_model(model_name: str,
                        train_text_counts: np.ndarray,
                        train_labels: np.ndarray,
                        test_text_counts: np.ndarray,
                        test_labels: np.ndarray,
                        average: str) -> BernoulliNB | MultinomialNB | MLPClassifier | SVC:
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

    # Predict on the test set
    test_preds = model.predict(test_text_counts)

    # Evaluate predictions
    precision, recall, f1, support = precision_recall_fscore_support(test_labels, test_preds, average=average)
    accuracy = accuracy_score(test_labels, test_preds)

    # Print scores
    print(f'{model_name} precision = {precision:.3f}')
    print(f'{model_name} recall = {recall:.3f}')
    print(f'{model_name} F1 = {f1:.3f}')
    print(f'{model_name} accuracy = {accuracy:.3f}\n')

    return model


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

"""
def run_mcts_on_text(text: str) -> None:
    print('Original\n')
    print(f'{text}\n')
    print(f'Stress = {stress_scoring_fn(text):.3f}')
    print(f'Entropy = {domain_entropy_scoring_fn(text):.3f}')

    print()

    # Run the MCTS search
    for context_dependent in [True, False]:
        if context_dependent:
            print('Context dependent')
        else:
            print('Context independent')

        mcts_nodes = mcts_explainer.explain(text=text, context_dependent=context_dependent)

        # Select the best MCTSNode
        best_mcts_node = get_best_mcts_node(mcts_nodes, max_percent_unmasked=0.5)
        words = best_mcts_node.words
        mask = best_mcts_node.mask

        # Print the result
        print('Masked\n')
        print(' '.join(word if mask_element == 1 else '<mask>' for word, mask_element in zip(words, mask)) + '\n')

        text_phrases = [' '.join(words[i] for i in phrase) for phrase in get_phrases(mask)]

        for text_phrase in text_phrases:
            print(f'{text_phrase}\n')

        print(f'Stress = {np.mean([stress_scoring_fn(text_phrase) for text_phrase in text_phrases]):.3f}')
        print(f'Entropy = {np.mean([domain_entropy_scoring_fn(text_phrase) for text_phrase in text_phrases]):.3f}')
        print(f'Percent unmasked = {100 * best_mcts_node.percent_unmasked:.2f}%')
"""


def run_mcts(train_embeddings_path: Path,
             test_embeddings_path: Path,
             model_name: Literal['bnb', 'mnb', 'mlp', 'svm'],
             save_dir: Path,
             alpha: float = 10.0) -> None:
    """Runs MCTS to extract context-dependent and context-independent explanations from text.

    :param train_embeddings_path: The path to the train embeddings pickle file, which include the train text.
    :param test_embeddings_path: The path to the test embeddings, which include the test text.
    :param model_name: The name of the model to train.
    :param save_dir: The path to a directory where the results will be saved.
    :param alpha: The value of the parameter that weighs context entropy compared to stress.
    """
    # Load train data and embeddings
    train_texts, train_embeddings, train_stress, train_subreddits = load_data_and_embeddings(
        embeddings_path=train_embeddings_path,
    )
    print(f'Train size = {len(train_texts):,}')
    print(f'Num stressed = {np.sum(train_stress):,}\n')

    # Load test data and embeddings
    test_texts, test_embeddings, test_stress, test_subreddits = load_data_and_embeddings(
        embeddings_path=test_embeddings_path,
    )
    print(f'Test size = {len(test_texts):,}')
    print(f'Num stressed = {np.sum(test_stress):,}\n')

    # Fit CountVectorizer
    stress_count_vectorizer = CountVectorizer(ngram_range=(1, 1)).fit(train_texts)

    # Convert train and test texts to counts
    train_text_counts = stress_count_vectorizer.transform(train_texts)
    test_text_counts = stress_count_vectorizer.transform(test_texts)

    # Train stress model
    print('Stress model')
    stress_model = train_sklearn_model(
        model_name=model_name,
        train_text_counts=train_text_counts,
        train_labels=train_stress,
        test_text_counts=test_text_counts,
        test_labels=test_stress,
        average='binary'
    )
    print()

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

    # Fit CountVectorizer
    context_count_vectorizer = CountVectorizer(ngram_range=(1, 1)).fit(train_texts_selected)

    # Convert train and test texts to counts
    train_text_selected_counts = context_count_vectorizer.transform(train_texts_selected)
    test_text_selected_counts = context_count_vectorizer.transform(test_texts_selected)

    # Train subreddit model
    print('Context (subreddit) model')
    context_model = train_sklearn_model(
        model_name=model_name,
        train_text_counts=train_text_selected_counts,
        train_labels=train_subreddits_selected,
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
        max_phrases=3,
        min_phrase_length=5,
        scoring_fn=stress_and_context_scoring_fn,
        n_rollout=20,
        min_percent_unmasked=0.2,
        c_puct=10.0,
        num_expand_nodes=10,
        high2low=False
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

        for context_dependent, masked_stress, masked_entropy in [(True, masked_stress_dependent, masked_entropy_dependent),
                                                                 (False, masked_stress_independent, masked_entropy_independent)]:
            mcts_nodes = mcts_explainer.explain(text=text, context_dependent=context_dependent)
            best_mcts_node = get_best_mcts_node(mcts_nodes, max_percent_unmasked=0.5)
            words = best_mcts_node.words
            mask = best_mcts_node.mask

            text_phrases = [' '.join(words[i] for i in phrase) for phrase in get_phrases(mask)]

            masked_stress.append(np.mean([stress_scoring_fn(text_phrase) for text_phrase in text_phrases]))
            masked_entropy.append(np.mean([context_entropy_scoring_fn(text_phrase) for text_phrase in text_phrases]))

    # Save the results
    results = {
        'texts': stressed_test_texts,
        'original_stress': np.array(original_stress),
        'masked_stress_dependent': np.array(masked_stress_dependent),
        'masked_stress_independent': np.array(masked_stress_independent),
        'original_entropy': np.array(original_entropy),
        'masked_entropy_dependent': np.array(masked_entropy_dependent),
        'masked_entropy_independent': np.array(masked_entropy_independent)
    }

    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / f'{model_name}_alpha_{alpha}.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    from tap import Tap

    class Args(Tap):
        train_embeddings_path: Path  # The path to the train embeddings pickle file, which include the train text.
        test_embeddings_path: Path  # The path to the test embeddings, which include the test text.
        model_name: Literal['bnb', 'mnb', 'mlp', 'svm']  # The name of the model to train.
        save_dir: Path  # The path to a directory where the results will be saved.
        alpha: float = 10.0  # The value of the parameter that weighs context entropy compared to stress.

    run_mcts(**Args().parse_args().as_dict())
