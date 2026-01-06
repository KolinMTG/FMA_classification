import tensorflow as tf
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
def add_true_labels(csv_path, label_mapping_path, output_csv_path) -> pd.DataFrame:
    """
    Docstring pour add_true_labels
    
    : csv_path: path to the input CSV file containing TFRecord file paths
    : label_mapping_path: path to the CSV file mapping label numbers to genres
    : output_csv_path: path to save the updated CSV with true labels
    """
    # Load CSV and label mapping
    df = pd.read_csv(csv_path)
    label_map = pd.read_csv(label_mapping_path)
    # Create a dictionary mapping label number to genre
    label_dict = dict(zip(label_map['label'], label_map['genre']))
    
    true_labels = []

    for tfrecord_path in df['file_path']:
        # Create a TFRecord dataset
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        # Define the features in the TFRecord
        feature_description = {
            'spectrogram': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
        }

        # Parse the TFRecord (assuming one example per file)
        for raw_record in raw_dataset.take(1):
            example = tf.io.parse_single_example(raw_record, feature_description)
            label_num = int(example['label'].numpy())
            true_labels.append(label_dict[label_num])

    # Add new column to the dataframe
    df['true_label'] = true_labels
    # Save updated CSV
    df.to_csv(output_csv_path, index=False)
    print(f"CSV saved with true_label at {output_csv_path}")
    return df

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class ModelPerformanceAnalyzer:
    """Comprehensive analysis of model predictions."""
    
    def __init__(self, csv_path: str, output_folder: str):
        """
        Initialize analyzer.
        
        Args:
            csv_path: Path to predictions CSV file
            output_folder: Directory to save all analysis outputs
        """
        self.csv_path = csv_path
        self.output_folder = output_folder
        self.df = None
        self.classes = None
        self.num_classes = None
        self.metrics = {}
        
        # Create output directory structure
        self.plots_dir = os.path.join(output_folder, "plots")
        self.tables_dir = os.path.join(output_folder, "tables")
        self.reports_dir = os.path.join(output_folder, "reports")
        
        for directory in [self.plots_dir, self.tables_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def load_data(self):
        """Load and preprocess the CSV data."""
        print("Loading data...")
        self.df = pd.read_csv(self.csv_path)
        
        # Filter only successful predictions
        self.df_success = self.df[self.df['status'] == 'success'].copy()
        
        # Get unique classes from true labels
        self.classes = sorted(self.df_success['true_label'].unique())
        self.num_classes = len(self.classes)
        
        print(f"Loaded {len(self.df)} total files")
        print(f"Successful predictions: {len(self.df_success)}")
        print(f"Failed predictions: {len(self.df) - len(self.df_success)}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.classes}")
    
    def compute_overall_metrics(self):
        """Compute overall classification metrics."""
        print("\nComputing overall metrics...")
        
        y_true = self.df_success['true_label']
        y_pred = self.df_success['predicted_label']
        
        # Overall metrics
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        self.metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        self.metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        self.metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        self.metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        self.metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # Per-class metrics
        self.metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.classes, zero_division=0, output_dict=True
        )
        
        # Confusion matrix
        self.metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=self.classes)
        
        print(f"Overall Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"Macro F1-Score: {self.metrics['f1_macro']:.4f}")
    
    def plot_confusion_matrices(self):
        """Generate confusion matrix plots (normalized and raw)."""
        print("\nGenerating confusion matrices...")
        
        cm = self.metrics['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Raw confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes,
                    yticklabels=self.classes, ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix_raw.png'))
        plt.close()
        
        # Normalized confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=self.classes,
                    yticklabels=self.classes, ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Proportion'})
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix_normalized.png'))
        plt.close()
    
    def analyze_class_distribution(self):
        """Analyze and visualize class distributions."""
        print("\nAnalyzing class distributions...")
        
        # True class distribution
        true_dist = self.df_success['true_label'].value_counts().reindex(self.classes, fill_value=0)
        pred_dist = self.df_success['predicted_label'].value_counts().reindex(self.classes, fill_value=0)
        
        # Plot distribution comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        true_dist.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_title('True Label Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        pred_dist.plot(kind='bar', ax=ax2, color='salmon', edgecolor='black')
        ax2.set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Class', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'class_distribution.png'))
        plt.close()
        
        # Save distribution table
        dist_df = pd.DataFrame({
            'True_Label_Count': true_dist,
            'Predicted_Label_Count': pred_dist
        })
        dist_df.to_csv(os.path.join(self.tables_dir, 'class_distribution.csv'))
    
    def analyze_per_class_performance(self):
        """Analyze performance metrics per class."""
        print("\nAnalyzing per-class performance...")
        
        report_dict = self.metrics['classification_report']
        
        # Extract per-class metrics
        per_class_metrics = []
        for class_name in self.classes:
            if class_name in report_dict:
                metrics = report_dict[class_name]
                per_class_metrics.append({
                    'Class': class_name,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score'],
                    'Support': metrics['support']
                })
        
        per_class_df = pd.DataFrame(per_class_metrics)
        per_class_df.to_csv(os.path.join(self.tables_dir, 'per_class_metrics.csv'), index=False)
        
        # Plot per-class metrics
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics_to_plot = ['Precision', 'Recall', 'F1-Score']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            per_class_df.plot(x='Class', y=metric, kind='bar', ax=ax, 
                             color='teal', edgecolor='black', legend=False)
            ax.set_title(f'{metric} per Class', fontsize=14, fontweight='bold')
            ax.set_xlabel('Class', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_ylim(0, 1.1)
            ax.tick_params(axis='x', rotation=45)
            
            # Map metric name to the actual key in self.metrics
            if metric == 'Precision':
                metric_key = 'precision_macro'
            elif metric == 'Recall':
                metric_key = 'recall_macro'
            elif metric == 'F1-Score':
                metric_key = 'f1_macro'
            else:
                metric_key = None
            
            if metric_key and metric_key in self.metrics:
                macro_value = self.metrics[metric_key]
                ax.axhline(y=macro_value, color='red', linestyle='--', 
                          label=f'Macro Avg: {macro_value:.3f}')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'per_class_metrics.png'))
        plt.close()
    
    def analyze_confidence_scores(self):
        """Analyze confidence score distributions."""
        print("\nAnalyzing confidence scores...")
        
        # Overall confidence distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.df_success['confidence'], bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(self.df_success['confidence'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df_success["confidence"].mean():.3f}')
        ax.axvline(self.df_success['confidence'].median(), color='green', linestyle='--', 
                   label=f'Median: {self.df_success["confidence"].median():.3f}')
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Overall Confidence Score Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'confidence_distribution_overall.png'))
        plt.close()
        
        # Confidence by predicted class (boxplot)
        fig, ax = plt.subplots(figsize=(12, 6))
        self.df_success.boxplot(column='confidence', by='predicted_label', ax=ax, patch_artist=True)
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('Confidence Score', fontsize=12)
        ax.set_title('Confidence Distribution by Predicted Class', fontsize=14, fontweight='bold')
        plt.suptitle('')  # Remove default title
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'confidence_by_predicted_class.png'))
        plt.close()
        
        # Confidence for correct vs incorrect predictions
        self.df_success['correct'] = self.df_success['true_label'] == self.df_success['predicted_label']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        correct_conf = self.df_success[self.df_success['correct']]['confidence']
        incorrect_conf = self.df_success[~self.df_success['correct']]['confidence']
        
        ax.hist([correct_conf, incorrect_conf], bins=30, label=['Correct', 'Incorrect'], 
                color=['green', 'red'], alpha=0.6, edgecolor='black')
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Confidence Distribution: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'confidence_correct_vs_incorrect.png'))
        plt.close()
        
        # Confidence statistics
        conf_stats = pd.DataFrame({
            'Overall': [self.df_success['confidence'].mean(), self.df_success['confidence'].std(),
                       self.df_success['confidence'].min(), self.df_success['confidence'].max()],
            'Correct': [correct_conf.mean(), correct_conf.std(), correct_conf.min(), correct_conf.max()],
            'Incorrect': [incorrect_conf.mean(), incorrect_conf.std(), incorrect_conf.min(), incorrect_conf.max()]
        }, index=['Mean', 'Std', 'Min', 'Max'])
        conf_stats.to_csv(os.path.join(self.tables_dir, 'confidence_statistics.csv'))
    
    def analyze_errors(self):
        """Perform detailed error analysis."""
        print("\nAnalyzing errors...")
        
        # Misclassified files
        misclassified = self.df_success[~self.df_success['correct']].copy()
        misclassified_summary = misclassified[['filename', 'true_label', 'predicted_label', 'confidence', 'num_segments']]
        misclassified_summary = misclassified_summary.sort_values('confidence', ascending=False)
        misclassified_summary.to_csv(os.path.join(self.tables_dir, 'misclassified_files.csv'), index=False)
        
        print(f"Total misclassified files: {len(misclassified)}")
        
        # Common confusion patterns
        confusion_pairs = misclassified.groupby(['true_label', 'predicted_label']).size().reset_index(name='count')
        confusion_pairs = confusion_pairs.sort_values('count', ascending=False)
        confusion_pairs.to_csv(os.path.join(self.tables_dir, 'confusion_patterns.csv'), index=False)
        
        # Plot top confusion pairs
        if len(confusion_pairs) > 0:
            top_confusions = confusion_pairs.head(10)
            fig, ax = plt.subplots(figsize=(12, 6))
            top_confusions['pair'] = top_confusions['true_label'] + ' → ' + top_confusions['predicted_label']
            ax.barh(top_confusions['pair'], top_confusions['count'], color='coral', edgecolor='black')
            ax.set_xlabel('Count', fontsize=12)
            ax.set_ylabel('True → Predicted', fontsize=12)
            ax.set_title('Top 10 Confusion Patterns', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'top_confusion_patterns.png'))
            plt.close()
    
    def analyze_segments(self):
        """Analyze segment-level predictions."""
        print("\nAnalyzing segment-level data...")
        
        # Distribution of number of segments
        fig, ax = plt.subplots(figsize=(10, 6))
        self.df_success['num_segments'].value_counts().sort_index().plot(kind='bar', ax=ax, 
                                                                          color='steelblue', edgecolor='black')
        ax.set_xlabel('Number of Segments', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Number of Segments per File', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'num_segments_distribution.png'))
        plt.close()
        
        # Confidence vs num_segments scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if c else 'red' for c in self.df_success['correct']]
        ax.scatter(self.df_success['num_segments'], self.df_success['confidence'], 
                  c=colors, alpha=0.5, s=30)
        ax.set_xlabel('Number of Segments', fontsize=12)
        ax.set_ylabel('Confidence Score', fontsize=12)
        ax.set_title('Confidence vs Number of Segments', fontsize=14, fontweight='bold')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Correct',
                                 markerfacecolor='g', markersize=8),
                          Line2D([0], [0], marker='o', color='w', label='Incorrect',
                                 markerfacecolor='r', markersize=8)]
        ax.legend(handles=legend_elements)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'confidence_vs_segments.png'))
        plt.close()
        
        # Segment-level prediction analysis
        segment_cols = [col for col in self.df_success.columns if col.startswith('predicted_label_')]
        confidence_cols = [col for col in self.df_success.columns if col.startswith('confidence_') 
                          and not col == 'confidence']
        
        if segment_cols:
            # Calculate segment consistency (how many segments agree with final prediction)
            def calculate_consistency(row):
                final_pred = row['predicted_label']
                segment_preds = [row[col] for col in segment_cols if pd.notna(row[col])]
                if not segment_preds:
                    return np.nan
                return sum(1 for pred in segment_preds if pred == final_pred) / len(segment_preds)
            
            self.df_success['segment_consistency'] = self.df_success.apply(calculate_consistency, axis=1)
            
            # Plot segment consistency distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(self.df_success['segment_consistency'].dropna(), bins=20, 
                   color='mediumpurple', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Segment Consistency (Proportion Agreeing with Final Prediction)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Segment Prediction Consistency', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'segment_consistency.png'))
            plt.close()
            
            # Segment consistency vs accuracy
            fig, ax = plt.subplots(figsize=(10, 6))
            correct_consistency = self.df_success[self.df_success['correct']]['segment_consistency'].dropna()
            incorrect_consistency = self.df_success[~self.df_success['correct']]['segment_consistency'].dropna()
            
            ax.hist([correct_consistency, incorrect_consistency], bins=15, 
                   label=['Correct', 'Incorrect'], color=['green', 'red'], 
                   alpha=0.6, edgecolor='black')
            ax.set_xlabel('Segment Consistency', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Segment Consistency: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'segment_consistency_by_correctness.png'))
            plt.close()
            
            # Average segment-level confidence
            if confidence_cols:
                self.df_success['avg_segment_confidence'] = self.df_success[confidence_cols].mean(axis=1)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(self.df_success['avg_segment_confidence'], self.df_success['confidence'],
                          c=colors, alpha=0.5, s=30)
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
                ax.set_xlabel('Average Segment-Level Confidence', fontsize=12)
                ax.set_ylabel('Final File-Level Confidence', fontsize=12)
                ax.set_title('Segment-Level vs File-Level Confidence', fontsize=14, fontweight='bold')
                ax.legend(handles=legend_elements)
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, 'segment_vs_file_confidence.png'))
                plt.close()
    
    def analyze_by_file_type(self):
        """Compare performance between audio and tfrecord files."""
        print("\nAnalyzing by file type...")
        
        if 'file_type' not in self.df_success.columns:
            print("No file_type column found, skipping file type analysis")
            return
        
        file_types = self.df_success['file_type'].unique()
        
        if len(file_types) < 2:
            print("Only one file type found, skipping comparative analysis")
            return
        
        # Performance by file type
        file_type_metrics = []
        for ftype in file_types:
            subset = self.df_success[self.df_success['file_type'] == ftype]
            y_true = subset['true_label']
            y_pred = subset['predicted_label']
            
            file_type_metrics.append({
                'File_Type': ftype,
                'Count': len(subset),
                'Accuracy': accuracy_score(y_true, y_pred),
                'F1_Macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'Mean_Confidence': subset['confidence'].mean(),
                'Mean_Num_Segments': subset['num_segments'].mean()
            })
        
        ft_df = pd.DataFrame(file_type_metrics)
        ft_df.to_csv(os.path.join(self.tables_dir, 'performance_by_file_type.csv'), index=False)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy
        ft_df.plot(x='File_Type', y='Accuracy', kind='bar', ax=axes[0, 0], 
                  color='skyblue', edgecolor='black', legend=False)
        axes[0, 0].set_title('Accuracy by File Type', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylim(0, 1.1)
        
        # F1 Score
        ft_df.plot(x='File_Type', y='F1_Macro', kind='bar', ax=axes[0, 1], 
                  color='lightcoral', edgecolor='black', legend=False)
        axes[0, 1].set_title('F1-Score (Macro) by File Type', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylim(0, 1.1)
        
        # Mean Confidence
        ft_df.plot(x='File_Type', y='Mean_Confidence', kind='bar', ax=axes[1, 0], 
                  color='lightgreen', edgecolor='black', legend=False)
        axes[1, 0].set_title('Mean Confidence by File Type', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylim(0, 1.1)
        
        # Mean Num Segments
        ft_df.plot(x='File_Type', y='Mean_Num_Segments', kind='bar', ax=axes[1, 1], 
                  color='wheat', edgecolor='black', legend=False)
        axes[1, 1].set_title('Mean Number of Segments by File Type', fontsize=12, fontweight='bold')
        
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'performance_by_file_type.png'))
        plt.close()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\nGenerating summary report...")
        
        report_path = os.path.join(self.reports_dir, 'analysis_summary_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.csv_path}\n\n")
            
            # Dataset overview
            f.write("-" * 80 + "\n")
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Files: {len(self.df)}\n")
            f.write(f"Successful Predictions: {len(self.df_success)}\n")
            f.write(f"Failed Predictions: {len(self.df) - len(self.df_success)}\n")
            f.write(f"Success Rate: {len(self.df_success) / len(self.df) * 100:.2f}%\n")
            f.write(f"Number of Classes: {self.num_classes}\n")
            f.write(f"Classes: {', '.join(self.classes)}\n\n")
            
            # Overall performance
            f.write("-" * 80 + "\n")
            f.write("OVERALL PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accuracy: {self.metrics['accuracy']:.4f} ({self.metrics['accuracy']*100:.2f}%)\n")
            f.write(f"Precision (Macro): {self.metrics['precision_macro']:.4f}\n")
            f.write(f"Precision (Micro): {self.metrics['precision_micro']:.4f}\n")
            f.write(f"Recall (Macro): {self.metrics['recall_macro']:.4f}\n")
            f.write(f"Recall (Micro): {self.metrics['recall_micro']:.4f}\n")
            f.write(f"F1-Score (Macro): {self.metrics['f1_macro']:.4f}\n")
            f.write(f"F1-Score (Micro): {self.metrics['f1_micro']:.4f}\n\n")
            
            # Per-class performance
            f.write("-" * 80 + "\n")
            f.write("PER-CLASS PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
            f.write("-" * 80 + "\n")
            
            report_dict = self.metrics['classification_report']
            for class_name in self.classes:
                if class_name in report_dict:
                    metrics = report_dict[class_name]
                    f.write(f"{class_name:<20} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                           f"{metrics['f1-score']:<12.4f} {int(metrics['support']):<10}\n")
            f.write("\n")
            
            # Confidence statistics
            f.write("-" * 80 + "\n")
            f.write("CONFIDENCE STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Overall Mean Confidence: {self.df_success['confidence'].mean():.4f}\n")
            f.write(f"Overall Median Confidence: {self.df_success['confidence'].median():.4f}\n")
            f.write(f"Overall Std Confidence: {self.df_success['confidence'].std():.4f}\n")
            
            if 'correct' in self.df_success.columns:
                correct_conf = self.df_success[self.df_success['correct']]['confidence']
                incorrect_conf = self.df_success[~self.df_success['correct']]['confidence']
                f.write(f"\nCorrect Predictions - Mean Confidence: {correct_conf.mean():.4f}\n")
                f.write(f"Incorrect Predictions - Mean Confidence: {incorrect_conf.mean():.4f}\n")
            f.write("\n")
            
            # Error analysis
            f.write("-" * 80 + "\n")
            f.write("ERROR ANALYSIS\n")
            f.write("-" * 80 + "\n")
            misclassified = self.df_success[~self.df_success['correct']]
            f.write(f"Total Misclassified: {len(misclassified)}\n")
            f.write(f"Error Rate: {len(misclassified) / len(self.df_success) * 100:.2f}%\n\n")
            
            if len(misclassified) > 0:
                confusion_pairs = misclassified.groupby(['true_label', 'predicted_label']).size()
                confusion_pairs = confusion_pairs.sort_values(ascending=False).head(10)
                f.write("Top 10 Confusion Patterns:\n")
                for (true_label, pred_label), count in confusion_pairs.items():
                    f.write(f"  {true_label} → {pred_label}: {count} files\n")
            f.write("\n")
            
            # Segment analysis
            f.write("-" * 80 + "\n")
            f.write("SEGMENT ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean Number of Segments: {self.df_success['num_segments'].mean():.2f}\n")
            f.write(f"Median Number of Segments: {self.df_success['num_segments'].median():.0f}\n")
            f.write(f"Min/Max Segments: {self.df_success['num_segments'].min():.0f} / {self.df_success['num_segments'].max():.0f}\n")
            
            if 'segment_consistency' in self.df_success.columns:
                f.write(f"\nMean Segment Consistency: {self.df_success['segment_consistency'].mean():.4f}\n")
                correct_consistency = self.df_success[self.df_success['correct']]['segment_consistency'].mean()
                incorrect_consistency = self.df_success[~self.df_success['correct']]['segment_consistency'].mean()
                f.write(f"Segment Consistency (Correct): {correct_consistency:.4f}\n")
                f.write(f"Segment Consistency (Incorrect): {incorrect_consistency:.4f}\n")
            f.write("\n")
            
            # File type analysis
            if 'file_type' in self.df_success.columns and len(self.df_success['file_type'].unique()) > 1:
                f.write("-" * 80 + "\n")
                f.write("FILE TYPE COMPARISON\n")
                f.write("-" * 80 + "\n")
                for ftype in self.df_success['file_type'].unique():
                    subset = self.df_success[self.df_success['file_type'] == ftype]
                    y_true = subset['true_label']
                    y_pred = subset['predicted_label']
                    acc = accuracy_score(y_true, y_pred)
                    f.write(f"\n{ftype.upper()}:\n")
                    f.write(f"  Count: {len(subset)}\n")
                    f.write(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
                    f.write(f"  Mean Confidence: {subset['confidence'].mean():.4f}\n")
                    f.write(f"  Mean Segments: {subset['num_segments'].mean():.2f}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Summary report saved to: {report_path}")
    
    def run_full_analysis(self):
        """Execute complete analysis pipeline."""
        print("\n" + "=" * 80)
        print("STARTING COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        self.load_data()
        self.compute_overall_metrics()
        self.plot_confusion_matrices()
        self.analyze_class_distribution()
        self.analyze_per_class_performance()
        self.analyze_confidence_scores()
        self.analyze_errors()
        self.analyze_segments()
        self.analyze_by_file_type()
        self.generate_summary_report()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nAll outputs saved to: {self.output_folder}")
        print(f"  - Plots: {self.plots_dir}")
        print(f"  - Tables: {self.tables_dir}")
        print(f"  - Reports: {self.reports_dir}")
        print("\n" + "=" * 80)



def main():
    """Main execution function."""
    # Configuration
    add_true_labels(
        csv_path=r"predictions/test_tfrecords/.predictions_summary.csv",
        label_mapping_path=r"data/metadata/label_mapping.csv",
        output_csv_path=r"predictions/test_tfrecords/.predictions_with_true_labels.csv"
    )

    
    csv_path = r"predictions/test_tfrecords/.predictions_with_true_labels.csv"
    output_folder = r"plots/model_performance_analysis"
    
    # Initialize and run analyzer
    analyzer = ModelPerformanceAnalyzer(csv_path, output_folder)
    analyzer.run_full_analysis()
    
    print("\n✓ Analysis package generated successfully!")
    print(f"✓ Review the summary report at: {os.path.join(output_folder, 'reports', 'analysis_summary_report.txt')}")


if __name__ == "__main__":
    main()