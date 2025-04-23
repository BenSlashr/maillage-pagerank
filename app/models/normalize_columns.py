"""
Module utilitaire pour normaliser les noms de colonnes dans les DataFrames.
"""

import pandas as pd
import logging
from typing import Tuple, Optional

def normalize_link_columns(links_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les noms de colonnes pour les liens (source/destination).
    
    Args:
        links_df: DataFrame contenant les liens
        
    Returns:
        DataFrame avec des noms de colonnes normalisés
    """
    # Vérifier et normaliser les noms de colonnes
    source_col = None
    dest_col = None
    
    # Rechercher les colonnes source/destination dans différents formats
    possible_source_cols = ['Source', 'source']
    possible_dest_cols = ['Destination', 'destination', 'target', 'Target']
    
    for col in possible_source_cols:
        if col in links_df.columns:
            source_col = col
            break
            
    for col in possible_dest_cols:
        if col in links_df.columns:
            dest_col = col
            break
    
    # Vérifier que les colonnes nécessaires existent
    if source_col is None or dest_col is None:
        logging.error(f"Colonnes source/destination introuvables. Colonnes disponibles: {', '.join(links_df.columns)}")
        return links_df
        
    # Créer une copie du DataFrame avec des noms de colonnes normalisés
    normalized_df = links_df.copy()
    if source_col != 'Source':
        normalized_df.rename(columns={source_col: 'Source'}, inplace=True)
        logging.info(f"Colonne '{source_col}' renommée en 'Source'")
    if dest_col != 'Destination':
        normalized_df.rename(columns={dest_col: 'Destination'}, inplace=True)
        logging.info(f"Colonne '{dest_col}' renommée en 'Destination'")
        
    return normalized_df
