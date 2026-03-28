import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_research_graphs(train_csv='checkpoints/train_metrics.csv', val_csv='checkpoints/val_metrics.csv'):
    # 1. Plot Training Dynamics
    if os.path.exists(train_csv):
        df_train = pd.read_csv(train_csv)
        df_train['GlobalStep'] = df_train.index # Continuous X-axis
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        color = 'tab:blue'
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Train Loss', color=color)
        # We use a rolling mean (window=5) to smooth out those "jumping" batches so you can see the true trend!
        ax1.plot(df_train['GlobalStep'], df_train['TrainLoss'].rolling(5, min_periods=1).mean(), color=color, linewidth=2, label='Smoothed Loss')
        ax1.plot(df_train['GlobalStep'], df_train['TrainLoss'], color=color, alpha=0.2, label='Raw Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Grad Norm', color=color)
        ax2.plot(df_train['GlobalStep'], df_train['GradNorm'], color=color, alpha=0.5, label='Grad Norm')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Training Dynamics (Loss & Gradients)')
        fig.tight_layout()
        plt.savefig('research_plot_train.png', dpi=300)
        print("Saved training plot to research_plot_train.png")

    # 2. Plot Validation Metrics
    if os.path.exists(val_csv):
        df_val = pd.read_csv(val_csv)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss Plot
        ax1.plot(df_val['Epoch'], df_val['TotalLoss'], marker='o', color='black', label='Total Val Loss')
        ax1.plot(df_val['Epoch'], df_val['FocalLoss'], linestyle='--', color='blue', label='Focal Loss')
        ax1.plot(df_val['Epoch'], df_val['RegLoss'], linestyle='--', color='green', label='Reg Loss')
        ax1.set_title('Validation Loss over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Recall Plot
        ax2.plot(df_val['Epoch'], df_val['Car_Recall3'] * 100, marker='s', color='blue', label='Car (IoU 0.3)')
        ax2.plot(df_val['Epoch'], df_val['Ped_Recall3'] * 100, marker='^', color='orange', label='Pedestrian (IoU 0.3)')
        ax2.plot(df_val['Epoch'], df_val['Cyc_Recall3'] * 100, marker='d', color='green', label='Cyclist (IoU 0.3)')
        ax2.set_title('Detection Recall over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Recall (%)')
        ax2.legend()
        ax2.grid(True)
        
        fig.tight_layout()
        plt.savefig('research_plot_val.png', dpi=300)
        print("Saved validation plot to research_plot_val.png")

if __name__ == '__main__':
    plot_research_graphs()