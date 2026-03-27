# plot_logs.py
import re
import matplotlib.pyplot as plt

def plot_training_logs(log_file='checkpoints/train.log'):
    steps, losses, grad_norms = [], [],[]
    
    with open(log_file, 'r') as f:
        for line in f:
            # Extract metrics using regex
            match = re.search(r'Step (\d+)/.*loss=([\d\.]+).*grad_norm=([\d\.]+)', line)
            if match:
                steps.append(len(steps)) # Continuous step counter
                losses.append(float(match.group(2)))
                grad_norms.append(float(match.group(3)))

    if not steps:
        print("No training data found in log yet.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(steps, losses, label='Train Loss', color='blue', alpha=0.7)
    ax1.set_title('Training Loss over Steps')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(steps, grad_norms, label='Gradient Norm', color='red', alpha=0.7)
    ax2.set_title('Gradient Norm over Steps (Target < 1.0)')
    ax2.set_ylabel('Grad Norm')
    ax2.set_xlabel('Training Steps')
    ax2.axhline(1.0, color='black', linestyle='--', label='max_norm limit')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_graphs.png', dpi=300)
    print("Saved graphs to training_graphs.png!")

if __name__ == '__main__':
    plot_training_logs()