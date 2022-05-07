"""Analyzes the MCTS explanations output by run_mcts.py in terms of stress and context entropy."""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


def analyze_mcts_results(results_dir: Path,
                         model_name: str,
                         alpha: float) -> None:
    """Analyzes the MCTS explanations output by run_mcts.py in terms of stress and context entropy.

    :param results_dir: Path to a directory containing the results from run_mcts.py. Plots will be saved here.
    :param model_name: The name of the model.
    :param alpha: The value of the parameter that weighs context entropy compared to stress.
    """
    # Load MCTS results
    with open(results_dir / f'{model_name}_alpha_{alpha}.pkl', 'rb') as f:
        results = pickle.load(f)

    # Extract MCTS results
    original_stress = results['original_stress']
    masked_stress_dependent = results['masked_stress_dependent']
    masked_stress_independent = results['masked_stress_independent']
    original_entropy = results['original_entropy']
    masked_entropy_dependent = results['masked_entropy_dependent']
    masked_entropy_independent = results['masked_entropy_independent']

    # Plot stress
    stress_bins = np.linspace(0, 1, 20)
    plt.clf()
    plt.figure(figsize=(12, 8))
    plt.hist(original_stress, stress_bins, alpha=0.5, label='Original')
    plt.hist(masked_stress_dependent, stress_bins, alpha=0.5, label='Context-Dependent')
    plt.hist(masked_stress_independent, stress_bins, alpha=0.5, label='Context-Independent')
    plt.legend(fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.yticks(fontsize=16)
    plt.xlabel('Stress Score', fontsize=20)
    plt.xticks(fontsize=16)
    plt.title(rf'Stress Score for Original Text and Explanations ($\alpha = {alpha}$)', fontsize=24)
    plt.savefig(results_dir / f'{model_name}_alpha_{alpha}_stress.pdf', bbox_inches='tight')

    # Plot entropy
    max_entropy = -np.log2(1 / 3)
    entropy_bins = np.linspace(0, max_entropy, 20)
    plt.clf()
    plt.figure(figsize=(12, 8))
    plt.hist(original_entropy, entropy_bins, alpha=0.5, label='Original')
    plt.hist(masked_entropy_dependent, entropy_bins, alpha=0.5, label='Context-Dependent')
    plt.hist(masked_entropy_independent, entropy_bins, alpha=0.5, label='Context-Independent')
    plt.legend(fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.yticks(fontsize=16)
    plt.xlabel('Context Entropy', fontsize=20)
    plt.xticks(fontsize=16)
    plt.title(rf'Context Entropy for Original Text and Explanations ($\alpha = {alpha}$)', fontsize=24)
    plt.savefig(results_dir / f'{model_name}_alpha_{alpha}_entropy.pdf', bbox_inches='tight')

    # Print stress and entropy results
    print(f'Average stress (original) = '
          f'{np.mean(original_stress):.3f} +/- {np.std(original_stress):.3f}')

    print(f'Average stress (dependent) = '
          f'{np.mean(masked_stress_dependent):.3f} +/- {np.std(masked_stress_dependent):.3f}')

    print(f'Average stress (independent) = '
          f'{np.mean(masked_stress_independent):.3f} +/- {np.std(masked_stress_independent):.3f}')

    print()

    print(f'Average entropy (original) = '
          f'{np.mean(original_entropy):.3f} +/- {np.std(original_entropy):.3f}')

    print(f'Average entropy (dependent) = '
          f'{np.mean(masked_entropy_dependent):.3f} +/- {np.std(masked_entropy_dependent):.3f}')

    print(f'Average entropy (independent) = '
          f'{np.mean(masked_entropy_independent):.3f} +/- {np.std(masked_entropy_independent):.3f}')


    # Compute stress and entropy diffs
    diff_stress_dependent_original = masked_stress_dependent - original_stress
    diff_stress_independent_original = masked_stress_independent - original_stress
    diff_stress_dependent_independent = masked_stress_dependent - masked_stress_independent

    diff_entropy_dependent_original = masked_entropy_dependent - original_entropy
    diff_entropy_independent_original = masked_entropy_independent - original_entropy
    diff_entropy_dependent_independent = masked_entropy_dependent - masked_entropy_independent

    # Print stress and entropy diffs
    print(f'Average difference in stress (dependent - original) = '
          f'{np.mean(diff_stress_dependent_original):.3f} +/- {np.std(diff_stress_dependent_original):.3f} '
          f'(p = {wilcoxon(masked_stress_dependent, original_stress).pvalue:.4e})')

    print(f'Average difference in stress (independent - original) = '
          f'{np.mean(diff_stress_independent_original):.3f} +/- {np.std(diff_stress_independent_original):.3f} '
          f'(p = {wilcoxon(masked_stress_independent, original_stress).pvalue:.4e})')

    print(f'Average difference in stress (dependent - independent) = '
          f'{np.mean(diff_stress_dependent_independent):.3f} +/- {np.std(diff_stress_dependent_independent):.3f} '
          f'(p = {wilcoxon(masked_stress_dependent, masked_stress_independent).pvalue:.4e})')

    print()

    print(f'Average difference in entropy (dependent - original) = '
          f'{np.mean(diff_entropy_dependent_original):.3f} +/- {np.std(diff_entropy_dependent_original):.3f} '
          f'(p = {wilcoxon(masked_entropy_dependent, original_entropy).pvalue:.4e})')

    print(f'Average difference in entropy (independent - original) = '
          f'{np.mean(diff_entropy_independent_original):.3f} +/- {np.std(diff_entropy_independent_original):.3f} '
          f'(p = {wilcoxon(masked_entropy_independent, original_entropy).pvalue:.4e})')

    print(f'Average difference in entropy (dependent - independent) = '
          f'{np.mean(diff_entropy_dependent_independent):.3f} +/- {np.std(diff_entropy_dependent_independent):.3f} '
          f'(p = {wilcoxon(masked_entropy_dependent, masked_entropy_independent).pvalue:.4e})')


if __name__ == '__main__':
    from tap import Tap

    class Args(Tap):
        results_dir: Path  # Path to a directory containing the results from run_mcts.py. Plots will be saved here.
        model_name: str  # The name of the model.
        alpha: float  # The value of the parameter that weighs context entropy compared to stress.

    analyze_mcts_results(**Args().parse_args().as_dict())
