#!/usr/bin/env python3
"""
Plot metrics from inspect-ai evaluation logs.

This script loads EvalLog files (both .json and .eval formats) and creates various plots
to visualize evaluation metrics like accuracy, token usage, and sample-level results.

Usage:
    python plot.py <log_file_or_directory> [options]
    
Examples:
    python plot.py logs/                              # Plot all logs in directory
    python plot.py logs/eval.json                    # Plot single JSON log
    python plot.py logs/*.eval --plot-type accuracy  # Plot accuracy for .eval files
"""

import argparse
import json
import zipfile
import glob
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass
class EvalLogSummary:
    """Summary of key metrics from an EvalLog"""
    file_path: str
    task_name: str
    model_name: str
    created_at: datetime
    status: str
    accuracy: Optional[float] = None
    total_samples: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    completion_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None


class EvalLogLoader:
    """Load and parse EvalLog files (JSON and .eval formats)"""
    
    @staticmethod
    def load_json_log(file_path: str) -> Dict[str, Any]:
        """Load a JSON format log file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def load_eval_log(file_path: str) -> Dict[str, Any]:
        """Load a .eval format log file (ZIP archive)"""
        try:
            with zipfile.ZipFile(file_path, 'r') as z:
                # Try to find the main eval data
                files = z.namelist()
                
                # Look for samples data first
                sample_files = [f for f in files if f.startswith('samples/') and f.endswith('.json')]
                
                # Start with basic structure
                eval_data = {
                    "version": 2,
                    "status": "success",
                    "samples": []
                }
                
                # Load sample data
                for sample_file in sample_files:
                    try:
                        with z.open(sample_file) as f:
                            sample_data = json.load(f)
                            eval_data["samples"].append(sample_data)
                    except Exception as e:
                        print(f"Warning: Could not load {sample_file}: {e}")
                
                # Try to load start.json for metadata
                if '_journal/start.json' in files:
                    try:
                        with z.open('_journal/start.json') as f:
                            start_data = json.load(f)
                            eval_data.update(start_data)
                    except Exception as e:
                        print(f"Warning: Could not load start.json: {e}")
                
                return eval_data
                
        except Exception as e:
            print(f"Error loading .eval file {file_path}: {e}")
            return {}
    
    @staticmethod
    def load_log(file_path: str) -> Dict[str, Any]:
        """Load any supported log format"""
        if file_path.endswith('.json'):
            return EvalLogLoader.load_json_log(file_path)
        elif file_path.endswith('.eval'):
            return EvalLogLoader.load_eval_log(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")


class EvalLogParser:
    """Parse EvalLog data into structured summaries"""
    
    @staticmethod
    def parse_log(log_data: Dict[str, Any], file_path: str) -> EvalLogSummary:
        """Parse log data into an EvalLogSummary"""
        
        # Extract basic info
        status = log_data.get("status", "unknown")
        created_str = log_data.get("eval", {}).get("created", log_data.get("created", ""))
        
        # Parse creation time
        created_at = None
        if created_str:
            try:
                # Handle ISO format with timezone
                created_at = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
            except:
                try:
                    created_at = datetime.strptime(created_str, "%Y-%m-%dT%H:%M:%S")
                except:
                    pass
        
        # Extract task and model info
        eval_info = log_data.get("eval", {})
        task_name = eval_info.get("task", eval_info.get("task_display_name", "unknown"))
        
        # Model name extraction
        model_name = "unknown"
        if "plan" in log_data:
            model_name = log_data["plan"].get("config", {}).get("model", "unknown")
        elif "model" in eval_info:
            model_name = eval_info["model"]
        
        # Calculate accuracy from results
        accuracy = None
        results = log_data.get("results")
        if results and "scores" in results:
            scores = results["scores"]
            if "accuracy" in scores:
                accuracy = scores["accuracy"].get("value")
        
        # Count samples
        samples = log_data.get("samples", [])
        total_samples = len(samples)
        
        # Extract token usage
        stats = log_data.get("stats", {})
        model_usage = stats.get("model_usage", {})
        
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        
        # Sum across all models used
        for model, usage in model_usage.items():
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0) 
            total_tokens += usage.get("total_tokens", 0)
        
        # Parse completion time and duration
        completion_time = None
        duration_seconds = None
        
        completed_str = stats.get("completed_at", "")
        if completed_str:
            try:
                completion_time = datetime.fromisoformat(completed_str.replace('Z', '+00:00'))
                if created_at and completion_time:
                    duration_seconds = (completion_time - created_at).total_seconds()
            except:
                pass
        
        # Extract error info
        error_message = None
        if status == "error":
            error_info = log_data.get("error", {})
            error_message = error_info.get("message", "Unknown error")
        
        return EvalLogSummary(
            file_path=file_path,
            task_name=task_name,
            model_name=model_name,
            created_at=created_at or datetime.now(),
            status=status,
            accuracy=accuracy,
            total_samples=total_samples,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            completion_time=completion_time,
            duration_seconds=duration_seconds,
            error_message=error_message
        )


class EvalLogPlotter:
    """Create various plots from EvalLog data"""
    
    def __init__(self, summaries: List[EvalLogSummary]):
        self.summaries = summaries
        self.df = pd.DataFrame([vars(s) for s in summaries])
        
    def plot_accuracy_over_time(self, save_path: Optional[str] = None):
        """Plot accuracy scores over time"""
        successful_runs = self.df[self.df['status'] == 'success'].copy()
        
        if successful_runs.empty:
            print("No successful runs found for accuracy plot")
            return
            
        successful_runs = successful_runs.dropna(subset=['accuracy'])
        
        if successful_runs.empty:
            print("No accuracy data found")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Group by model for different colors
        for model in successful_runs['model_name'].unique():
            model_data = successful_runs[successful_runs['model_name'] == model]
            plt.scatter(model_data['created_at'], model_data['accuracy'], 
                       label=model, alpha=0.7, s=50)
        
        plt.xlabel('Time')
        plt.ylabel('Accuracy')
        plt.title('Evaluation Accuracy Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Accuracy plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_token_usage(self, save_path: Optional[str] = None):
        """Plot token usage statistics"""
        valid_runs = self.df[self.df['total_tokens'] > 0].copy()
        
        if valid_runs.empty:
            print("No token usage data found")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Input vs Output tokens scatter
        ax1.scatter(valid_runs['input_tokens'], valid_runs['output_tokens'], 
                   alpha=0.6, s=50)
        ax1.set_xlabel('Input Tokens')
        ax1.set_ylabel('Output Tokens')
        ax1.set_title('Input vs Output Token Usage')
        ax1.grid(True, alpha=0.3)
        
        # Total tokens over time
        valid_runs_time = valid_runs.dropna(subset=['created_at'])
        if not valid_runs_time.empty:
            ax2.scatter(valid_runs_time['created_at'], valid_runs_time['total_tokens'],
                       alpha=0.6, s=50)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Total Tokens')
            ax2.set_title('Token Usage Over Time')
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Token usage plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_model_comparison(self, save_path: Optional[str] = None):
        """Plot model performance comparison"""
        successful_runs = self.df[(self.df['status'] == 'success') & 
                                 (self.df['accuracy'].notna())].copy()
        
        if successful_runs.empty:
            print("No data available for model comparison")
            return
        
        # Group by model and calculate stats
        model_stats = successful_runs.groupby('model_name')['accuracy'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mean accuracy by model
        ax1.bar(range(len(model_stats)), model_stats['mean'], 
                yerr=model_stats['std'], capsize=5, alpha=0.7)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Mean Accuracy')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(range(len(model_stats)))
        ax1.set_xticklabels(model_stats['model_name'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Sample count by model
        ax2.bar(range(len(model_stats)), model_stats['count'], alpha=0.7)
        ax2.set_xlabel('Model') 
        ax2.set_ylabel('Number of Evaluations')
        ax2.set_title('Evaluation Count by Model')
        ax2.set_xticks(range(len(model_stats)))
        ax2.set_xticklabels(model_stats['model_name'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_success_rate(self, save_path: Optional[str] = None):
        """Plot success rate over time"""
        if self.df.empty:
            print("No data available for success rate plot")
            return
            
        # Group by hour and calculate success rate
        self.df['hour'] = self.df['created_at'].dt.floor('H')
        hourly_stats = self.df.groupby('hour').agg({
            'status': lambda x: (x == 'success').mean(),
            'file_path': 'count'
        }).reset_index()
        hourly_stats.columns = ['hour', 'success_rate', 'total_runs']
        
        plt.figure(figsize=(12, 6))
        plt.plot(hourly_stats['hour'], hourly_stats['success_rate'], 
                marker='o', linewidth=2, markersize=6)
        plt.xlabel('Time')
        plt.ylabel('Success Rate')
        plt.title('Evaluation Success Rate Over Time')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Success rate plot saved to {save_path}")
        else:
            plt.show()
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n=== Evaluation Log Summary ===")
        print(f"Total evaluations: {len(self.summaries)}")
        
        status_counts = self.df['status'].value_counts()
        print(f"Status breakdown:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
        
        successful_runs = self.df[self.df['status'] == 'success']
        if not successful_runs.empty:
            print(f"\nSuccessful runs: {len(successful_runs)}")
            
            if successful_runs['accuracy'].notna().any():
                mean_acc = successful_runs['accuracy'].mean()
                print(f"Average accuracy: {mean_acc:.3f}")
            
            total_tokens = successful_runs['total_tokens'].sum()
            if total_tokens > 0:
                print(f"Total tokens used: {total_tokens:,}")
        
        # Model breakdown
        model_counts = self.df['model_name'].value_counts()
        print(f"\nModels used:")
        for model, count in model_counts.items():
            print(f"  {model}: {count}")
        
        # Time range
        if not self.df['created_at'].isna().all():
            start_time = self.df['created_at'].min()
            end_time = self.df['created_at'].max()
            print(f"\nTime range: {start_time} to {end_time}")


def find_log_files(path: str) -> List[str]:
    """Find all log files in given path"""
    path_obj = Path(path)
    
    if path_obj.is_file():
        return [str(path_obj)]
    elif path_obj.is_dir():
        # Find all .json and .eval files
        json_files = glob.glob(os.path.join(path, "*.json"))
        eval_files = glob.glob(os.path.join(path, "*.eval"))
        return sorted(json_files + eval_files)
    else:
        # Handle glob patterns
        return sorted(glob.glob(path))


def main():
    parser = argparse.ArgumentParser(
        description="Plot metrics from inspect-ai evaluation logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "path", 
        help="Path to log file, directory, or glob pattern"
    )
    parser.add_argument(
        "--plot-type", 
        choices=["all", "accuracy", "tokens", "models", "success"],
        default="all",
        help="Type of plot to generate (default: all)"
    )
    parser.add_argument(
        "--output-dir", 
        help="Directory to save plots (default: show plots)"
    )
    parser.add_argument(
        "--summary-only", 
        action="store_true",
        help="Only print summary, don't generate plots"
    )
    
    args = parser.parse_args()
    
    # Find log files
    log_files = find_log_files(args.path)
    if not log_files:
        print(f"No log files found at {args.path}")
        return
    
    print(f"Loading {len(log_files)} log files...")
    
    # Load and parse logs
    summaries = []
    for file_path in log_files:
        try:
            log_data = EvalLogLoader.load_log(file_path)
            if log_data:
                summary = EvalLogParser.parse_log(log_data, file_path)
                summaries.append(summary)
                print(f" {file_path}")
            else:
                print(f" {file_path} - failed to load")
        except Exception as e:
            print(f" {file_path} - error: {e}")
    
    if not summaries:
        print("No valid log files loaded")
        return
    
    # Create plotter
    plotter = EvalLogPlotter(summaries)
    
    # Print summary
    plotter.print_summary()
    
    if args.summary_only:
        return
    
    # Generate plots
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    plot_types = ["accuracy", "tokens", "models", "success"] if args.plot_type == "all" else [args.plot_type]
    
    for plot_type in plot_types:
        save_path = None
        if output_dir:
            save_path = os.path.join(output_dir, f"{plot_type}_plot.png")
        
        if plot_type == "accuracy":
            plotter.plot_accuracy_over_time(save_path)
        elif plot_type == "tokens":
            plotter.plot_token_usage(save_path)
        elif plot_type == "models":
            plotter.plot_model_comparison(save_path)
        elif plot_type == "success":
            plotter.plot_success_rate(save_path)


if __name__ == "__main__":
    main()