import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # For the legend

def plot_policy(policy_probs: list):
    n_states = len(policy_probs)
    
    # --- Plot 2: Probabilistic Policy Map (All Actions) ---
    plt.figure(figsize=(12, 3))
    plt.xlim(-0.7, n_states - 0.3)
    plt.ylim(-0.6, 0.6)
    plt.yticks([])
    plt.xticks(range(n_states))
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)

    
    # Define scaling constants
    ARROW_LENGTH = 0.45
    HEAD_WIDTH = 0.15
    STAY_DOT_MAX = 35  # Max markersize for 100% stay probability
    
    for s in range(n_states):
        # Get probabilities for each action in this state
        p_left, p_stay, p_right = policy_probs[s]
        
        # 1. LEFT Action (Red Arrow)
        if p_left > 0.01: # Only plot if probability is significant
            plt.arrow(s, 0, -ARROW_LENGTH * p_left, 0, 
                      head_width=HEAD_WIDTH * p_left, 
                      head_length=0.1 * p_left, 
                      fc='red', ec='red', alpha=0.6)
        
        # 2. STAY Action (Green Dot)
        if p_stay > 0.01:
            plt.plot(s, 0, 'go', markersize=STAY_DOT_MAX * p_stay, alpha=0.5)
            
        # 3. RIGHT Action (Blue Arrow)
        if p_right > 0.01:
            plt.arrow(s, 0, ARROW_LENGTH * p_right, 0, 
                      head_width=HEAD_WIDTH * p_right, 
                      head_length=0.1 * p_right, 
                      fc='blue', ec='blue', alpha=0.6)
    
    # Create a clear legend for the different markers
    legend_elements = [
        Line2D([0], [0], color='red', marker='<', linestyle='None', markersize=10, label='Left'),
        Line2D([0], [0], color='green', marker='o', linestyle='None', markersize=10, label='Stay'),
        Line2D([0], [0], color='blue', marker='>', linestyle='None', markersize=10, label='Right')
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               ncol=3, frameon=False)
    
    plt.title('Policy Distribution across States\n(Size/Length proportional to Action Probability)')
    plt.tight_layout()
    plt.show()