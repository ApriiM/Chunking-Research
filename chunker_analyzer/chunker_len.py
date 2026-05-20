import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Ścieżki do danych
PIRB_DATA_DIR = "../data/pirb/pirb_data"
RESULTS_DIR = "../data/pirb/results"
PROCESSED_DIR = "../data/processed"

def load_metadata(exp_name):
    """Wczytuje metadata.json dla eksperymentu"""
    metadata_path = Path(PIRB_DATA_DIR) / exp_name / "metadata.json"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_results(exp_name):
    """Wczytuje wyniki dla eksperymentu"""
    results_path = Path(RESULTS_DIR) / exp_name
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_avg_chunk_length(exp_name):
    """Oblicza średnią długość chunka na podstawie passages.jsonl"""
    passages_path = Path(PIRB_DATA_DIR) / exp_name / "passages" / "passages.jsonl"
    total_length = 0
    count = 0
    
    with open(passages_path, 'r', encoding='utf-8') as f:
        for line in f:
            passage = json.loads(line)
            total_length += len(passage['contents'])
            count += 1
    
    return total_length / count if count > 0 else 0

def collect_all_data():
    """Zbiera dane ze wszystkich eksperymentów"""
    data = defaultdict(lambda: defaultdict(list))
    
    # Iteruj po wszystkich eksperymentach w pirb_data
    pirb_data_path = Path(PIRB_DATA_DIR)
    if not pirb_data_path.exists():
        raise FileNotFoundError(f"Katalog {PIRB_DATA_DIR} nie istnieje")
    
    for exp_dir in pirb_data_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        
        try:
            # Wczytaj metadata
            metadata = load_metadata(exp_name)
            dataset_slug = metadata['dataset_slug']
            chunker_name = metadata['chunker_name']
            chunk_count = metadata['chunk_count']
            
            # Wczytaj wyniki
            results = load_results(exp_name)
            metrics = results['metrics']
            
            # Oblicz średnią długość chunka
            avg_chunk_length = calculate_avg_chunk_length(exp_name)
            
            # Całkowita długość wszystkich chunków
            total_length = avg_chunk_length * chunk_count
            
            # Zapisz dane
            data[dataset_slug][chunker_name] = {
                'avg_chunk_length': avg_chunk_length,
                'chunk_count': chunk_count,
                'total_length': total_length,
                'accuracy_1': metrics['Accuracy@1'],
                'accuracy_5': metrics['Accuracy@1-5'][4],  # Accuracy@5
                'recall_10': metrics['Recall@10'],
                'mrr_10': metrics['MRR@10'],
                'ndcg_10': metrics['NDCG@10']
            }
            
        except Exception as e:
            print(f"Błąd przetwarzania {exp_name}: {e}")
            continue
    
    return data

def create_visualizations(data):
    """Tworzy 3 wykresy (dla każdej metryki chunk-owej)"""
    
    # Przygotuj kolory dla chunkerów
    all_chunkers = set()
    for dataset_data in data.values():
        all_chunkers.update(dataset_data.keys())
    
    chunker_colors = {}
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_chunkers)))
    for i, chunker in enumerate(sorted(all_chunkers)):
        chunker_colors[chunker] = colors[i]
    
    datasets = sorted(data.keys())
    n_datasets = len(datasets)
    
    # Metryki do wizualizacji
    metric_names = ['accuracy_1', 'accuracy_5', 'recall_10', 'mrr_10', 'ndcg_10']
    metric_labels = ['Accuracy@1', 'Accuracy@5', 'Recall@10', 'MRR@10', 'NDCG@10']
    
    # Parametry chunk-owe
    chunk_params = ['avg_chunk_length', 'chunk_count', 'total_length']
    chunk_labels = ['Średnia długość chunka (znaki)', 
                    'Liczba chunków', 
                    'Całkowita długość chunków (znaki)']
    
    # Twórz 3 wykresy (po jednym dla każdego parametru chunk-owego)
    for param_idx, (param, param_label) in enumerate(zip(chunk_params, chunk_labels)):
        
        fig, axes = plt.subplots(n_datasets, 5, figsize=(20, 4 * n_datasets))
        if n_datasets == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Wpływ: {param_label} na metryki', fontsize=16, y=0.995)
        
        for row, dataset in enumerate(datasets):
            dataset_data = data[dataset]
            
            for col, (metric, metric_label) in enumerate(zip(metric_names, metric_labels)):
                ax = axes[row, col]
                
                # Przygotuj dane do wykresu
                x_values = []
                y_values = []
                colors_list = []
                labels_list = []
                
                for chunker, chunker_data in dataset_data.items():
                    x_values.append(chunker_data[param])
                    y_values.append(chunker_data[metric])
                    colors_list.append(chunker_colors[chunker])
                    labels_list.append(chunker)
                
                # Rysuj punkty
                for x, y, color, label in zip(x_values, y_values, colors_list, labels_list):
                    ax.scatter(x, y, c=[color], s=100, alpha=0.7, edgecolors='black', 
                              linewidth=1, label=label)
                
                # Konfiguracja osi
                ax.set_ylim(0, 100)
                ax.set_xlabel(param_label, fontsize=9)
                ax.set_ylabel(metric_label, fontsize=9)
                ax.grid(True, alpha=0.3)
                
                # Tytuł tylko w pierwszym wierszu
                if row == 0:
                    ax.set_title(metric_label, fontsize=11, fontweight='bold')
                
                # Dataset name na osi y tylko w pierwszej kolumnie
                if col == 0:
                    ax.text(-0.3, 0.5, dataset, transform=ax.transAxes,
                           rotation=90, va='center', ha='center', 
                           fontsize=11, fontweight='bold')
        
        # Legenda
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=chunker_colors[chunker], 
                             markersize=8, label=chunker, markeredgecolor='black')
                  for chunker in sorted(all_chunkers)]
        
        fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5),
                  title='Chunkery', fontsize=9)
        
        plt.tight_layout(rect=[0, 0, 0.95, 0.99])
        
        # Zapisz wykres
        output_path = f'/mnt/user-data/outputs/visualization_{param}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano: {output_path}")
        plt.close()

def main():
    print("Zbieranie danych...")
    data = collect_all_data()
    
    print(f"Znaleziono {len(data)} datasetów:")
    for dataset, chunkers in data.items():
        print(f"  - {dataset}: {len(chunkers)} chunkerów")
    
    print("\nTworzenie wizualizacji...")
    create_visualizations(data)
    
    print("\nGotowe!")

if __name__ == "__main__":
    main()