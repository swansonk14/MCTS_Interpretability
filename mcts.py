"""Contains the classes that perform the MCTS for explanating text."""
import math
from functools import partial
from random import Random
from typing import Callable, List, Tuple

import numpy as np


class MCTSNode:
    """A node in a Monte Carlo Tree Search representing one or more phrases in the text."""

    def __init__(self,
                 words: Tuple[str],
                 mask: Tuple[int],
                 c_puct: float,
                 W: float = 0.0,
                 N: int = 0,
                 P: float = 0.0) -> None:
        """Initializes the MCTSNode object.

        :param words: The text of the example broken into a list of words.
        :param mask: A mask with 1s indicating words to use and 0s indicating words to ignore.
        :param c_puct: The hyperparameter that encourages exploration.
        :param W: The sum of the node value.
        :param N: The number of times of arrival at this node.
        :param P: The property score (reward) of this node.
        """
        self.words = words
        self.mask = mask
        self.mask_array = np.array(mask)
        self.c_puct = c_puct
        self.W = W
        self.N = N
        self.P = P
        self.children: List[MCTSNode] = []

    def Q(self) -> float:
        """Value that encourages exploitation of nodes with high reward."""
        return self.W / self.N if self.N > 0 else 0.0

    def U(self, n: int) -> float:
        """Value that encourages exploration of nodes with few visits."""
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)

    @property
    def percent_unmasked(self) -> float:
        """Return the percent of words that are unmasked."""
        return np.mean(self.mask_array)


def get_phrases(mask: Tuple[int]) -> List[List[int]]:
    """Gets the contiguous phrases as a list of lists of indices from a mask array.

    :param mask: A mask with 1s indicating words to use and 0s indicating words to ignore.
    :return: A list containing phrases where each phrase is a list of indices that are in the phrase.
    """
    phrases = []
    current_phrase = []
    expanded_mask = list(mask) + [0]

    for i, mask_element in enumerate(expanded_mask):
        if mask_element == 1:
            current_phrase.append(i)
        elif len(current_phrase) > 0:
            phrases.append(current_phrase)
            current_phrase = []

    return phrases


def text_score(words: Tuple[str], mask: Tuple[int], scoring_fn: Callable[[str], float]) -> float:
    """Computes the score of the masked text.

    :param words: The text of the example broken into a list of words.
    :param mask: A mask with 1s indicating words to use and 0s indicating words to ignore.
    :param scoring_fn: A function that takes as input a string and returns a score.
    :return: The score of the model applied to the subgraph induced by the provided coalition of nodes.
    """
    phrases = get_phrases(mask)
    text_phrases = [' '.join(words[i] for i in phrase) for phrase in phrases]
    scores = [scoring_fn(text_phrase) for text_phrase in text_phrases]
    score = np.mean(scores)

    return score


def get_best_mcts_node(results: List[MCTSNode], max_percent_unmasked: float) -> MCTSNode:
    """Get the MCTSNode with the highest reward (and smallest percent unmasked if tied)
       that has at most max_percent_unmasked unmasked words.

    :param results: A list of MCTSNodes.
    :param max_percent_unmasked: The maximum percent of words that are unmasked.
    :return: The MCTSNode with the highest reward (and smallest percent unmasked if tied)
             that has at most max_percent_unmasked unmasked word.
    """
    # Filter results to only include those with at most max_percent_unmasked unmasked words
    results = [result for result in results if result.percent_unmasked <= max_percent_unmasked]

    # Check that there exists a result with at most max_percent_unmasked unmasked words
    if len(results) == 0:
        raise ValueError(f'All results have more than {max_percent_unmasked} percent of words unmasked.')

    # Sort results by percent unmasked in case of a tie (max picks the first one it sees, so the smaller one)
    results = sorted(results, key=lambda result: result.percent_unmasked)

    # Find the result with the highest reward and break ties by preferring a less unmasked result
    best_result = max(results, key=lambda result: (result.P, -result.percent_unmasked))

    return best_result


class MCTS:
    """A class that runs Monte Carlo Tree Search to find explanatory phrases from a piece of text."""

    def __init__(self,
                 text: str,
                 max_phrases: int,
                 min_phrase_length: int,
                 scoring_fn: Callable[[str], float],
                 n_rollout: int,
                 min_percent_unmasked: float,
                 c_puct: float,
                 num_expand_nodes: int) -> None:
        """Creates the Monte Carlo Tree Search (MCTS) object.

        :param text: The input text.
        :param max_phrases: Maximum number of phrases in an explanation.
        :param min_phrase_length: Minimum number of words in a phrase.
        :param scoring_fn: A function that takes as input a string and returns a score.
        :param n_rollout: The number of times to build the Monte Carlo tree.
        :param min_percent_unmasked: The minimum percent of words unmasked,
                                     used as a stopping point for leaf nodes in the search tree.
        :param c_puct: The hyperparameter that encourages exploration.
        :param num_expand_nodes: The number of MCTS nodes to expand when extending the child nodes in the search tree.
        """
        self.text = text
        self.words = tuple(text.split())
        self.max_phrases = max_phrases
        self.min_phrase_length = min_phrase_length
        self.scoring_fn = scoring_fn
        self.n_rollout = n_rollout
        self.min_percent_unmasked = min_percent_unmasked
        self.c_puct = c_puct
        self.num_expand_nodes = num_expand_nodes
        self.random = Random(0)

        self.root_mask = tuple([1] * len(self.words))
        self.MCTSNodeClass = partial(MCTSNode, words=self.words, c_puct=self.c_puct)
        self.root = self.MCTSNodeClass(mask=self.root_mask)
        self.state_map = {self.root.mask: self.root}

    def mcts_rollout(self, tree_node: MCTSNode) -> float:
        """Performs a Monte Carlo Tree Search rollout.

        :param tree_node: An MCTSNode representing the root of the MCTS search.
        :return: The value (reward) of the rollout.
        """
        # Get a list of currently existing phrases (list of lists of indices)
        phrases = get_phrases(tree_node.mask)

        # Filter to only include phrases that are above the minimum phrase length
        shrinkable_phrases = [phrase for phrase in phrases if len(phrase) > self.min_phrase_length]

        # Stop the search and call this a leaf node if the node can't be expanded further
        if tree_node.percent_unmasked <= self.min_percent_unmasked or len(shrinkable_phrases) == 0:
            return tree_node.P

        # Expand if this node has never been visited
        if len(tree_node.children) == 0:
            # Maintain a set of all the masks added as children of the tree
            tree_children_masks = set()

            # Include left and right ends of each phrase as options to mask
            indices_to_mask = [i for phrase in shrinkable_phrases for i in [phrase[0], phrase[-1]]]

            # If not yet at max number of phrases, allow breaking a phrase
            # (as long as the resulting phrases are long enough)
            if len(phrases) < self.max_phrases:
                indices_to_mask += [i for phrase in shrinkable_phrases for i in
                                    phrase[self.min_phrase_length:-self.min_phrase_length]]

            # Limit the number of MCTS nodes to expand
            self.random.shuffle(indices_to_mask)
            indices_to_mask = indices_to_mask[:self.num_expand_nodes]

            # For each index, prune (mask) it and create a new MCTSNode (if it doesn't already exist)
            for index_to_mask in indices_to_mask:
                # Create new mask with additional index masked
                new_mask = list(tree_node.mask)
                new_mask[index_to_mask] = 0
                new_mask = tuple(new_mask)

                # Check the state map and merge with an existing mask if available
                new_node = self.state_map.setdefault(new_mask, self.MCTSNodeClass(mask=new_mask))

                # Add the mask to the children of the tree
                if new_mask not in tree_children_masks:
                    tree_node.children.append(new_node)
                    tree_children_masks.add(new_mask)

            # For each child in the tree, compute its reward
            for child in tree_node.children:
                if child.P == 0:
                    child.P = text_score(words=child.words, mask=child.mask, scoring_fn=self.scoring_fn)

        # Select the best child node and unroll it
        sum_count = sum(child.N for child in tree_node.children)
        selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(n=sum_count))
        v = self.mcts_rollout(tree_node=selected_node)
        selected_node.W += v
        selected_node.N += 1

        return v

    def run_mcts(self) -> List[MCTSNode]:
        """Runs the Monte Carlo Tree search.

        :return: A list of MCTSNode objects representing text masks sorted from highest to
                 smallest reward (for ties, the less unmasked result is first).
        """
        for _ in range(self.n_rollout):
            self.mcts_rollout(tree_node=self.root)

        explanations = [node for _, node in self.state_map.items()]

        # Sort by highest reward and break ties by preferring less unmasked results
        explanations = sorted(explanations, key=lambda x: (x.P, -x.percent_unmasked), reverse=True)

        return explanations


class MCTSExplainer:
    """A class that creates MCTS instances to find explanatory phrases in text."""

    def __init__(self,
                 max_phrases: int,
                 min_phrase_length: int,
                 scoring_fn: Callable[[str], float],
                 n_rollout: int,
                 min_percent_unmasked: float,
                 c_puct: float,
                 num_expand_nodes: int) -> None:
        """Creates the Monte Carlo Tree Search (MCTS) Explainer object.

        :param max_phrases: Maximum number of phrases in an explanation.
        :param min_phrase_length: Minimum number of words in a phrase.
        :param scoring_fn: A function that takes as input a string and returns a score.
        :param n_rollout: The number of times to build the Monte Carlo tree.
        :param min_percent_unmasked: The minimum percent of words unmasked,
                                     used as a stopping point for leaf nodes in the search tree.
        :param c_puct: The hyperparameter that encourages exploration.
        :param num_expand_nodes: The number of MCTS nodes to expand when extending the child nodes in the search tree.
        """
        self.max_phrases = max_phrases
        self.min_phrase_length = min_phrase_length
        self.scoring_fn = scoring_fn
        self.n_rollout = n_rollout
        self.min_percent_unmasked = min_percent_unmasked
        self.c_puct = c_puct
        self.num_expand_nodes = num_expand_nodes

    def explain(self, text: str, context_dependent: bool) -> List[MCTSNode]:
        return MCTS(
            text=text,
            max_phrases=self.max_phrases,
            min_phrase_length=self.min_phrase_length,
            scoring_fn=partial(self.scoring_fn, context_dependent=context_dependent),
            n_rollout=self.n_rollout,
            min_percent_unmasked=self.min_percent_unmasked,
            c_puct=self.c_puct,
            num_expand_nodes=self.num_expand_nodes
        ).run_mcts()
