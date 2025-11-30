# src/visualization.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def plot_heatmap(corr_matrix, labels, title='Correlation Heatmap', cmap='coolwarm', save_path=None):
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig



def plot_bar(categories, counts, title='Bar Chart', xlabel='Category', ylabel='Count', save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(categories, counts, color=plt.cm.viridis(np.linspace(0, 1, len(counts))))
    

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_roc(fpr, tpr, auc_score, title='ROC Curve', save_path=None):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_title(title)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

def plot_confusion_matrix(cm, labels=['Existing', 'Attrited'], title='Confusion Matrix', save_path=None):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

def plot_feature_importance(coefs, feature_names, title='Feature Importance (abs coef)', n_top=15, save_path=None):
    abs_coef = np.abs(coefs)
    idx = np.argsort(-abs_coef)[-n_top:]
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(idx))
    plt.barh(y_pos, abs_coef[idx], color='red', edgecolor='black')
    plt.yticks(y_pos, np.array(feature_names)[idx], fontsize=11)
    plt.xlabel('Absolute Coefficient', fontsize=12)
    plt.title(title or f'Top {n_top} Most Important Features', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig





