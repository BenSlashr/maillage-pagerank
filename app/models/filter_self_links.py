"""
Module pour filtrer les auto-liens dans les suggestions de maillage.
"""

import pandas as pd
import logging
from urllib.parse import urlparse

def normalize_url(url: str) -> str:
    """
    Normalise une URL pour les comparaisons.
    
    Args:
        url: URL à normaliser
        
    Returns:
        URL normalisée
    """
    # Supprimer les paramètres d'URL
    url = url.split('?')[0].split('#')[0]
    
    # Supprimer le slash final s'il existe
    if url.endswith('/'):
        url = url[:-1]
        
    # Convertir en minuscules
    url = url.lower()
    
    return url

def filter_self_links(suggestions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre les auto-liens dans un DataFrame de suggestions.
    
    Args:
        suggestions_df: DataFrame contenant les suggestions de liens
        
    Returns:
        DataFrame filtré sans auto-liens
    """
    if suggestions_df.empty:
        return suggestions_df
    
    # Vérifier que les colonnes nécessaires existent
    required_columns = ['source_url', 'target_url']
    if not all(col in suggestions_df.columns for col in required_columns):
        logging.warning("Les colonnes source_url et target_url sont requises pour filtrer les auto-liens")
        return suggestions_df
    
    # Nombre de suggestions avant filtrage
    initial_count = len(suggestions_df)
    
    # Créer des colonnes normalisées pour la comparaison
    suggestions_df['source_url_norm'] = suggestions_df['source_url'].apply(normalize_url)
    suggestions_df['target_url_norm'] = suggestions_df['target_url'].apply(normalize_url)
    
    # Filtrer les auto-liens
    filtered_df = suggestions_df[suggestions_df['source_url_norm'] != suggestions_df['target_url_norm']]
    
    # Supprimer les colonnes temporaires
    filtered_df = filtered_df.drop(columns=['source_url_norm', 'target_url_norm'])
    
    # Nombre de suggestions après filtrage
    final_count = len(filtered_df)
    removed_count = initial_count - final_count
    
    if removed_count > 0:
        logging.info(f"Filtrage des auto-liens: {removed_count} suggestions supprimées")
    
    return filtered_df
