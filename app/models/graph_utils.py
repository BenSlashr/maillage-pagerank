"""
Utilitaires pour la visualisation et l'analyse de graphes
"""
import logging
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any, Optional
import pandas as pd
from queue import Queue

def normalize_link_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les noms de colonnes pour les liens (Source/Destination)
    
    Args:
        df: DataFrame contenant les liens
        
    Returns:
        DataFrame avec les colonnes normalisées
    """
    # Copier le DataFrame pour éviter de modifier l'original
    df_copy = df.copy()
    
    # Mapper les noms de colonnes possibles vers les noms standards
    column_mapping = {
        'source': 'Source',
        'source_url': 'Source',
        'from': 'Source',
        'origin': 'Source',
        'destination': 'Destination',
        'target': 'Destination',
        'target_url': 'Destination',
        'to': 'Destination',
        'dest': 'Destination'
    }
    
    # Appliquer le mapping
    df_copy.rename(columns={col: column_mapping[col.lower()] 
                           for col in df_copy.columns 
                           if col.lower() in column_mapping}, 
                 inplace=True)
    
    return df_copy

def calculate_page_depths(links_df: pd.DataFrame, root_url: str) -> Dict[str, int]:
    """
    Calcule la profondeur de chaque page à partir d'une racine en utilisant un BFS
    
    Args:
        links_df: DataFrame contenant les liens (Source, Destination)
        root_url: URL de la page racine
        
    Returns:
        Dictionnaire avec les URLs comme clés et leur profondeur comme valeurs
    """
    # Normaliser les noms de colonnes
    links_df = normalize_link_columns(links_df)
    
    # Construire un graphe dirigé à partir des liens
    graph = defaultdict(list)
    for _, row in links_df.iterrows():
        source = row.get('Source')
        target = row.get('Destination')
        if isinstance(source, str) and isinstance(target, str):
            graph[source].append(target)
    
    # Initialiser les profondeurs
    depths = {root_url: 0}
    
    # File pour le BFS
    queue = Queue()
    queue.put(root_url)
    visited = {root_url}
    
    # Parcourir le graphe en largeur
    while not queue.empty():
        current_url = queue.get()
        current_depth = depths[current_url]
        
        for neighbor in graph.get(current_url, []):
            if neighbor not in visited:
                visited.add(neighbor)
                depths[neighbor] = current_depth + 1
                queue.put(neighbor)
    
    # Pour les URLs non accessibles depuis la racine, assigner une profondeur maximale
    all_urls = set()
    for _, row in links_df.iterrows():
        source = row.get('Source')
        target = row.get('Destination')
        if isinstance(source, str):
            all_urls.add(source)
        if isinstance(target, str):
            all_urls.add(target)
    
    max_depth = max(depths.values()) if depths else 0
    for url in all_urls:
        if url not in depths:
            depths[url] = max_depth + 1
    
    return depths

def identify_frequent_links(links_df: pd.DataFrame, threshold_percentage: float = 80.0) -> List[Tuple[str, str]]:
    """
    Identifie les liens qui apparaissent fréquemment dans le site (probablement menu/footer)
    
    Args:
        links_df: DataFrame contenant les liens (Source, Destination)
        threshold_percentage: Pourcentage de pages à partir duquel un lien est considéré comme fréquent
        
    Returns:
        Liste de tuples (source, destination) des liens fréquents
    """
    # Normaliser les noms de colonnes
    links_df = normalize_link_columns(links_df)
    
    # Compter les occurrences de chaque destination par source
    link_counts = Counter()
    for _, row in links_df.iterrows():
        source = row.get('Source')
        target = row.get('Destination')
        if isinstance(source, str) and isinstance(target, str):
            link_counts[(source, target)] += 1
    
    # Compter le nombre de pages sources uniques
    unique_sources = set(links_df['Source'].dropna())
    total_sources = len(unique_sources)
    
    # Identifier les liens qui apparaissent sur un pourcentage élevé de pages
    threshold = (threshold_percentage / 100.0) * total_sources
    frequent_links = []
    
    # Compter combien de sources pointent vers chaque destination
    destination_counts = Counter()
    for _, row in links_df.iterrows():
        target = row.get('Destination')
        if isinstance(target, str):
            destination_counts[target] += 1
    
    # Identifier les destinations qui sont liées depuis de nombreuses sources
    frequent_destinations = {dest for dest, count in destination_counts.items() 
                            if count >= threshold}
    
    # Créer la liste des liens fréquents
    for _, row in links_df.iterrows():
        source = row.get('Source')
        target = row.get('Destination')
        if isinstance(source, str) and isinstance(target, str):
            if target in frequent_destinations:
                frequent_links.append((source, target))
    
    return frequent_links
