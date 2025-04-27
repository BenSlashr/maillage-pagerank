"""
Module pour l'analyse SEO et la génération de suggestions de maillage interne.
Utilise un modèle BERT pour calculer la similarité sémantique entre les pages.
"""

import os
import re
import json
import logging
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Configuration du modèle BERT
MODEL_NAME = "distiluse-base-multilingual-cased-v1"

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Liste de stopwords français
STOP_WORDS = {"le", "la", "les", "un", "une", "des", "du", "de", "à", "au", "aux", "et", "ou", "mais", "donc", "car", "pour", "par", "sur", "dans", "en", "avec", "sans", "ce", "cette", "ces", "mon", "ton", "son", "notre", "votre", "leur", "mes", "tes", "ses", "nos", "vos", "leurs", "qui", "que", "quoi", "dont", "où", "comment", "pourquoi", "quand", "est", "sont", "sera", "seront", "était", "étaient", "été", "avoir", "eu", "être", "faire", "fait", "plus", "moins", "très", "trop", "peu", "beaucoup", "aussi", "ainsi", "comme", "même", "tout", "tous", "toute", "toutes", "autre", "autres", "non", "oui", "pas", "ne", "ni", "si", "alors", "après", "avant", "pendant", "depuis", "jusqu", "vers", "chez", "entre", "contre", "parmi", "selon", "suivant", "sous", "sur", "hors", "malgré", "sauf", "via"}

class SEOAnalyzer:
    """
    Classe pour analyser le contenu SEO et générer des suggestions de maillage interne.
    Utilise un modèle BERT pour calculer la similarité sémantique entre les pages.
    """
    
    def __init__(self, progress_callback: Optional[Callable[[str, int, int], None]] = None):
        """
        Initialise l'analyseur SEO avec le modèle BERT
        
        Args:
            progress_callback: Fonction de callback pour suivre la progression
        """
        self.progress_callback = progress_callback
        
        # Initialiser le modèle BERT
        try:
            self.model = SentenceTransformer(MODEL_NAME)
            device_name = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(device_name)
            logging.info(f"Utilisation du périphérique PyTorch: {device_name}")
            logging.info(f"Modèle SentenceTransformer chargé: {MODEL_NAME}")
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise e
    
    async def analyze(
        self,
        content_file: str,
        links_file: Optional[str] = None,
        gsc_file: Optional[str] = None,
        min_similarity: float = 0.2,
        anchor_suggestions: int = 3,
        linking_rules: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None,
        priority_urls: Optional[List[str]] = None,
        priority_urls_strict: bool = False
    ) -> str:
        """
        Analyse le contenu et génère des suggestions de maillage interne
        
        Args:
            content_file: Chemin vers le fichier Excel contenant le contenu des pages
            links_file: Chemin vers le fichier Excel contenant les liens existants (optionnel)
            gsc_file: Chemin vers le fichier Excel contenant les données GSC (optionnel)
            min_similarity: Score minimum de similarité pour les suggestions
            anchor_suggestions: Nombre de suggestions d'ancres
            linking_rules: Règles de maillage par segment (optionnel)
            priority_urls: Liste d'URL prioritaires (optionnel)
            priority_urls_strict: Mode strict pour les URL prioritaires
            
        Returns:
            Chemin vers le fichier Excel contenant les suggestions
        """
        start_time = time.time()
        
        # Charger les données
        logging.info("Chargement des données...")
        if self.progress_callback:
            await self.progress_callback("Chargement des données...", 0, 100)
            
        # Normaliser les chemins de fichiers pour s'assurer qu'ils sont absolus
        content_file = self._normalize_file_path(content_file)
        if links_file:
            links_file = self._normalize_file_path(links_file)
        if gsc_file:
            gsc_file = self._normalize_file_path(gsc_file)
            
        # Vérifier l'existence des fichiers avant de continuer
        if not os.path.exists(content_file):
            error_msg = f"Fichier de contenu introuvable: {content_file}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Charger le contenu des pages
        logging.info(f"Chargement du fichier de contenu: {content_file}")
        content_df = pd.read_excel(content_file)
        
        # Normaliser les noms de colonnes pour accepter différents formats
        # Mapper les noms de colonnes alternatifs vers les noms attendus
        column_mapping = {
            # Mappings standard
            "Adresse": "url",
            "URL": "url",
            "Titre": "title",
            "Title": "title",
            "Title 1": "title",  # Screaming Frog
            "Contenu": "content1",
            "Content": "content1",
            "Type": "type",
            "Type de contenu": "type",  # Screaming Frog
            "Segments": "type"  # Utiliser la colonne Segments comme type si disponible
        }
        
        # Détection des formats de fichiers (Screaming Frog ou crawler intégré)
        is_screaming_frog = False
        is_internal_crawler = False
        
        # Détection du crawler intégré
        if "Adresse" in content_df.columns and "Extracteur 1 1" in content_df.columns and "Titre" in content_df.columns and "H1" in content_df.columns:
            is_internal_crawler = True
            logging.info("Détection d'un export du crawler intégré")
            
            # Créer une colonne de contenu combinée pour les exports du crawler intégré
            content_columns = []
            
            # Ajouter le titre s'il existe
            if "Titre" in content_df.columns:
                content_columns.append("Titre")
                
            # Ajouter la balise H1 si elle existe
            if "H1" in content_df.columns:
                content_columns.append("H1")
                
            # Ajouter l'extracteur de contenu si disponible
            if "Extracteur 1 1" in content_df.columns:
                content_columns.append("Extracteur 1 1")
                logging.info("Utilisation de la colonne 'Extracteur 1 1' pour enrichir le contenu")
            
            # Combiner toutes ces colonnes en une seule colonne de contenu
            if content_columns:
                logging.info(f"Création d'une colonne de contenu combinée à partir de: {', '.join(content_columns)}")
                content_df['content1'] = content_df[content_columns].apply(
                    lambda row: ' '.join([str(val) for val in row if pd.notna(val) and str(val).strip() != '']), 
                    axis=1
                )
            
            # Si la colonne Type n'existe pas, utiliser Segments
            if "Segments" in content_df.columns:
                content_df['type'] = content_df['Segments']
            else:
                # Créer une colonne type par défaut si nécessaire
                content_df['type'] = "page"
                
        # Détection de Screaming Frog
        elif "Title 1" in content_df.columns and ("H1-1" in content_df.columns or any(col.startswith("H1-") for col in content_df.columns)):
            is_screaming_frog = True
            logging.info("Détection d'un export Screaming Frog")
            
            # Créer une colonne de contenu combinée pour les exports Screaming Frog
            content_columns = []
            
            # Ajouter le titre s'il existe
            if "Title 1" in content_df.columns:
                content_columns.append("Title 1")
                
            # Ajouter les balises H1 s'ils existent
            h1_columns = [col for col in content_df.columns if col.startswith("H1-") and not col.startswith("H1-Longueur")]
            content_columns.extend(h1_columns)
            
            # Ajouter les balises H2 s'ils existent
            h2_columns = [col for col in content_df.columns if col.startswith("H2-") and not col.startswith("H2-Longueur")]
            content_columns.extend(h2_columns)
            
            # Ajouter la meta description si elle existe
            if "Meta Description 1" in content_df.columns:
                content_columns.append("Meta Description 1")
                
            # Ajouter l'extracteur de contenu si disponible (contient souvent le contenu principal de la page)
            if "Extracteur 1 1" in content_df.columns:
                content_columns.append("Extracteur 1 1")
                logging.info("Utilisation de la colonne 'Extracteur 1 1' pour enrichir le contenu")
            
            # Combiner toutes ces colonnes en une seule colonne de contenu
            if content_columns:
                logging.info(f"Création d'une colonne de contenu combinée à partir de: {', '.join(content_columns)}")
                content_df['content1'] = content_df[content_columns].apply(
                    lambda row: ' '.join([str(val) for val in row if pd.notna(val) and str(val).strip() != '']), 
                    axis=1
                )
            
            # Si la colonne Type n'existe pas, utiliser Segments ou Type de contenu
            if "Segments" in content_df.columns:
                content_df['type'] = content_df['Segments']
            elif "Type de contenu" in content_df.columns:
                content_df['type'] = content_df['Type de contenu']
            else:
                # Créer une colonne type par défaut si nécessaire
                content_df['type'] = "page"
        
        # Renommer les colonnes si nécessaire
        for alt_col, std_col in column_mapping.items():
            if alt_col in content_df.columns and std_col not in content_df.columns:
                content_df.rename(columns={alt_col: std_col}, inplace=True)
                logging.info(f"Colonne '{alt_col}' renommée en '{std_col}'")
        
        # Vérifier les colonnes requises après normalisation
        required_columns = ["url", "title", "content1", "type"]
        missing_columns = [col for col in required_columns if col not in content_df.columns]
        
        if missing_columns:
            # Afficher les colonnes disponibles pour faciliter le débogage
            available_columns = ", ".join(content_df.columns)
            raise ValueError(f"Colonnes requises manquantes dans le fichier de contenu: {', '.join(missing_columns)}. Colonnes disponibles: {available_columns}")
        
        # Charger les liens existants si disponibles
        existing_links_df = None
        if links_file:
            if os.path.exists(links_file):
                logging.info(f"Chargement des liens existants: {links_file}")
                existing_links_df = pd.read_excel(links_file)
            else:
                logging.warning(f"Fichier de liens introuvable: {links_file}")
            
            # Normaliser les noms de colonnes pour le fichier de liens
            links_column_mapping = {
                "Source": "source",
                "URL source": "source",
                "URL Source": "source",
                "Adresse source": "source",
                "Target": "target",
                "Cible": "target",
                "URL cible": "target",
                "URL Cible": "target",
                "Adresse cible": "target",
                "Destination": "target"
            }
            
            # Vérifier si les colonnes Source/Destination existent déjà
            if "Source" in existing_links_df.columns and "Destination" in existing_links_df.columns:
                logging.info("Détection des colonnes Source/Destination dans le fichier de liens")
                existing_links_df.rename(columns={"Source": "source", "Destination": "target"}, inplace=True)
            
            # Renommer les colonnes si nécessaire
            for alt_col, std_col in links_column_mapping.items():
                if alt_col in existing_links_df.columns and std_col not in existing_links_df.columns:
                    existing_links_df.rename(columns={alt_col: std_col}, inplace=True)
                    logging.info(f"Colonne '{alt_col}' renommée en '{std_col}' dans le fichier de liens")
            
            # Vérifier les colonnes requises après normalisation
            required_columns = ["source", "target"]
            missing_columns = [col for col in required_columns if col not in existing_links_df.columns]
            
            if missing_columns:
                # Afficher les colonnes disponibles pour faciliter le débogage
                available_columns = ", ".join(existing_links_df.columns)
                raise ValueError(f"Colonnes requises manquantes dans le fichier de liens: {', '.join(missing_columns)}. Colonnes disponibles: {available_columns}")
        
        # Charger les données GSC si disponibles
        gsc_data = None
        if gsc_file:
            if os.path.exists(gsc_file):
                logging.info(f"Chargement des données GSC: {gsc_file}")
                try:
                    gsc_data = pd.read_excel(gsc_file)
                    
                    # Normaliser les noms de colonnes pour le fichier GSC
                    gsc_column_mapping = {
                        "Page": "page",
                        "URL": "page",
                        "Adresse": "page",
                        "Landing Page": "page",
                        "Landing page": "page",
                        "Query": "query",
                        "Requête": "query",
                        "Mot-clé": "query",
                        "Mot clé": "query",
                        "Keyword": "query",
                        "Clicks": "clicks",
                        "Clics": "clicks",
                        "Impressions": "impressions",
                        "Position": "position",
                        "Pos.": "position",
                        "Pos": "position",
                        "CTR": "ctr"
                    }
                    
                    # Renommer les colonnes si nécessaire
                    for alt_col, std_col in gsc_column_mapping.items():
                        if alt_col in gsc_data.columns and std_col not in gsc_data.columns:
                            gsc_data.rename(columns={alt_col: std_col}, inplace=True)
                            logging.info(f"Colonne '{alt_col}' renommée en '{std_col}' dans le fichier GSC")
                    
                    # Vérifier les colonnes requises après normalisation
                    required_columns = ["page", "query", "clicks", "impressions", "position"]
                    missing_columns = [col for col in required_columns if col not in gsc_data.columns]
                    
                    if missing_columns:
                        # Afficher les colonnes disponibles pour faciliter le débogage
                        available_columns = ", ".join(gsc_data.columns)
                        logging.warning(f"Colonnes manquantes dans le fichier GSC: {', '.join(missing_columns)}. Colonnes disponibles: {available_columns}")
                        logging.warning("Analyse sans données GSC.")
                        gsc_data = None
                        
                except Exception as e:
                    logging.error(f"Erreur lors du chargement du fichier GSC: {str(e)}")
                    logging.warning("Analyse sans données GSC.")
                    gsc_data = None
            else:
                logging.warning(f"Fichier GSC introuvable: {gsc_file}")
                logging.warning("Analyse sans données GSC.")
        
        # Traiter les données GSC
        gsc_queries = {}
        gsc_stats = {}
        if gsc_data is not None:
            logging.info("Traitement des données GSC...")
            if self.progress_callback:
                await self.progress_callback("Traitement des données GSC...", 5, 100)
            
            # Regrouper les requêtes par page
            for _, row in gsc_data.iterrows():
                page = row["page"]
                query = row["query"]
                clicks = row["clicks"]
                impressions = row["impressions"]
                position = row["position"]
                
                if page not in gsc_queries:
                    gsc_queries[page] = []
                    gsc_stats[page] = {"clics": 0, "impressions": 0, "position": 0}
                
                gsc_queries[page].append({
                    "query": query,
                    "clics": clicks,
                    "impressions": impressions,
                    "position": position
                })
                
                # Agréger les statistiques
                gsc_stats[page]["clics"] += clicks
                gsc_stats[page]["impressions"] += impressions
                # Moyenne pondérée pour la position
                if gsc_stats[page]["impressions"] > 0:
                    gsc_stats[page]["position"] = (gsc_stats[page]["position"] * (gsc_stats[page]["impressions"] - impressions) + position * impressions) / gsc_stats[page]["impressions"]
        
        # Générer les suggestions de liens
        logging.info("Génération des suggestions de liens...")
        if self.progress_callback:
            await self.progress_callback("Génération des suggestions de liens...", 10, 100)
        
        # Créer un ensemble d'URL prioritaires pour une recherche plus rapide
        priority_urls_set = set(priority_urls) if priority_urls else set()
        
        # Générer les suggestions de liens
        suggestions_df = self._generate_suggestions(
            content_df,
            existing_links_df,
            min_similarity,
            anchor_suggestions,
            linking_rules,
            priority_urls_set,
            priority_urls_strict,
            gsc_queries,
            gsc_stats
        )
        
        # Sauvegarder les résultats
        logging.info("Sauvegarde des résultats...")
        if self.progress_callback:
            await self.progress_callback("Sauvegarde des résultats...", 90, 100)
        
        result_file = self._save_results(suggestions_df)
        
        # Terminer
        elapsed_time = time.time() - start_time
        logging.info(f"Analyse terminée en {elapsed_time:.2f} secondes. Résultats sauvegardés dans: {result_file}")
        if self.progress_callback:
            await self.progress_callback(f"Analyse terminée en {elapsed_time:.2f} secondes", 100, 100)
        
        return result_file
    
    def _normalize_url(self, url: str) -> str:
        """Normalise une URL pour les comparaisons"""
        # Supprimer le protocole
        url = re.sub(r'^https?://', '', url)
        # Supprimer les paramètres et fragments
        url = re.sub(r'[?#].*$', '', url)
        # Supprimer les slashes de fin
        url = re.sub(r'/+$', '', url)
        # Convertir en minuscules
        url = url.lower()
        return url
        
    def _normalize_file_path(self, file_path: str) -> str:
        """Normalise un chemin de fichier pour s'assurer qu'il est absolu"""
        if not file_path:
            return file_path
            
        # Si le chemin est déjà absolu, le retourner tel quel
        if os.path.isabs(file_path):
            return file_path
            
        # Si le chemin commence par 'app/', s'assurer qu'il est relatif au répertoire racine
        if file_path.startswith('app/'):
            # Obtenir le répertoire racine (2 niveaux au-dessus du répertoire du module)
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            return os.path.join(root_dir, file_path)
            
        # Sinon, considérer le chemin comme relatif au répertoire courant
        return os.path.abspath(file_path)
    
    def _generate_suggestions(
        self,
        content_df: pd.DataFrame,
        existing_links_df: Optional[pd.DataFrame],
        min_similarity: float,
        anchor_suggestions: int,
        linking_rules: Optional[Dict[str, Dict[str, Dict[str, int]]]],
        priority_urls_set: Set[str],
        priority_urls_strict: bool,
        gsc_queries: Dict[str, List[Dict[str, Any]]],
        gsc_stats: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Génère des suggestions de liens internes en tenant compte des URL prioritaires.
        
        Args:
            content_df: DataFrame contenant le contenu des pages
            existing_links_df: DataFrame contenant les liens existants
            min_similarity: Score minimum de similarité pour les suggestions
            anchor_suggestions: Nombre de suggestions d'ancres
            linking_rules: Règles de maillage par segment
            priority_urls_set: Ensemble d'URL prioritaires
            priority_urls_strict: Mode strict pour les URL prioritaires
            gsc_queries: Requêtes GSC par page
            gsc_stats: Statistiques GSC par page
            
        Returns:
            DataFrame des suggestions de liens
        """
        # Créer un dictionnaire pour stocker les suggestions
        suggestions = []
        
        # Créer un ensemble de liens existants pour une recherche plus rapide
        existing_links_set = set()
        normalized_existing_links_set = set()  # Ensemble avec URLs normalisées
        bidirectional_links_set = set()  # Ensemble pour vérifier les liens dans les deux sens
        if existing_links_df is not None:
            # Vérifier les noms de colonnes disponibles
            if "source" in existing_links_df.columns and "target" in existing_links_df.columns:
                for _, row in existing_links_df.iterrows():
                    source = row["source"]
                    target = row["target"]
                    existing_links_set.add((source, target))
                    # Ajouter également une version normalisée pour la comparaison
                    source_norm = self._normalize_url(source)
                    target_norm = self._normalize_url(target)
                    normalized_existing_links_set.add((source_norm, target_norm))
                    # Ajouter les deux directions pour vérifier les liens bidirectionnels
                    bidirectional_links_set.add((source_norm, target_norm))
                    bidirectional_links_set.add((target_norm, source_norm))
            elif "Source" in existing_links_df.columns and "Destination" in existing_links_df.columns:
                for _, row in existing_links_df.iterrows():
                    source = row["Source"]
                    target = row["Destination"]
                    existing_links_set.add((source, target))
                    # Ajouter également une version normalisée pour la comparaison
                    source_norm = self._normalize_url(source)
                    target_norm = self._normalize_url(target)
                    normalized_existing_links_set.add((source_norm, target_norm))
                    # Ajouter les deux directions pour vérifier les liens bidirectionnels
                    bidirectional_links_set.add((source_norm, target_norm))
                    bidirectional_links_set.add((target_norm, source_norm))
            else:
                logging.warning("Colonnes de liens introuvables dans le fichier de liens. Colonnes disponibles: " + ", ".join(existing_links_df.columns))
            
            logging.info(f"Nombre de liens existants chargés: {len(existing_links_set)}")
        
        # Calculer les embeddings pour toutes les pages
        logging.info("Calcul des embeddings...")
        
        # Extraire le contenu principal pour chaque page
        contents = content_df["content1"].tolist()
        
        # Calculer les embeddings
        embeddings = self.model.encode(contents, show_progress_bar=True)
        
        # Calculer la matrice de similarité
        logging.info("Calcul de la matrice de similarité...")
        similarity_matrix = np.zeros((len(content_df), len(content_df)))
        
        for i in range(len(content_df)):
            for j in range(len(content_df)):
                # Calculer la similarité cosinus entre les embeddings
                similarity_matrix[i, j] = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
        
        # Calculer les similarités et générer les suggestions pour chaque page
        for i, (source_idx, source_row) in enumerate(content_df.iterrows()):
            source_url = source_row["url"]
            source_content = source_row["content1"]
            source_type = source_row["type"]
            
            # Mise à jour de la progression
            if self.progress_callback:
                # Créer une tâche asynchrone pour le callback sans attendre
                if asyncio.iscoroutinefunction(self.progress_callback):
                    asyncio.create_task(self.progress_callback(f"Génération des suggestions pour {source_url}", i, len(content_df)))
                else:
                    self.progress_callback(f"Génération des suggestions pour {source_url}", i, len(content_df))
            
            # Initialiser la liste des suggestions pour cette page
            suggestions_for_page = []
            
            # Si des règles de maillage sont définies, préparer un dictionnaire pour suivre le nombre de liens par type
            type_link_counts = {}
            if linking_rules and source_type in linking_rules:
                # Initialiser le compteur pour chaque type cible
                for target_type in linking_rules[source_type].keys():
                    # Compter le nombre de liens existants de ce type
                    existing_count = sum(1 for s, d in existing_links_set if s == source_url and content_df[content_df["url"] == d]["type"].iloc[0] == target_type)
                    type_link_counts[target_type] = {
                        "existing": existing_count,
                        "suggested": 0,
                        "min": linking_rules[source_type][target_type].get("min_links", 0),
                        "max": linking_rules[source_type][target_type].get("max_links", 10)
                    }
            
            # Créer deux listes de suggestions : une pour les suggestions obligatoires (pour atteindre le minimum par type)
            # et une pour les suggestions facultatives (basées uniquement sur la similarité)
            mandatory_suggestions = []
            optional_suggestions = []
            
            # Trier les indices par score de similarité (du plus élevé au plus bas)
            sorted_indices = np.argsort(similarity_matrix[source_idx])[::-1]
            
            # Bonus de similarité pour les URL prioritaires (pour les favoriser dans le tri)
            priority_bonus = 0.1 if priority_urls_set else 0
            
            # En mode strict, on limite les liens sortants des URL prioritaires
            # pour éviter de diluer leur PageRank
            if priority_urls_strict and source_url in priority_urls_set:
                logging.debug(f"Mode strict activé pour l'URL prioritaire: {source_url}")
                # On réduit le nombre de suggestions pour les URL prioritaires
                max_suggestions = 2  # Limiter à 2 liens sortants en mode strict
            else:
                max_suggestions = 10  # Valeur par défaut
            
            # Première passe : traiter les suggestions pour atteindre le nombre minimum de liens par type
            if linking_rules and source_type in linking_rules:
                for j in sorted_indices:
                    # Ignorer la page elle-même
                    if i == j:
                        continue
                    
                    # Récupérer les informations de la page cible
                    target_row = content_df.iloc[j]
                    target_url = target_row["url"]
                    target_type = target_row["type"]
                    
                    # Vérifier si ce type de cible est dans les règles et si le minimum n'est pas atteint
                    if target_type in linking_rules[source_type]:
                        # Récupérer les compteurs pour ce type
                        type_counts = type_link_counts.get(target_type, {})
                        if not type_counts:
                            continue
                            
                        # Vérifier si nous avons déjà atteint le minimum requis pour ce type
                        total_links = type_counts["existing"] + type_counts["suggested"]
                        if total_links >= type_counts["min"]:
                            continue
                            
                        # Vérifier si nous avons déjà atteint le maximum pour ce type
                        if total_links >= type_counts["max"]:
                            continue
                    
                        # Ignorer la page elle-même (par URL normalisée)
                        source_url_normalized = self._normalize_url(source_url)
                        target_url_normalized = self._normalize_url(target_url)
                        if source_url_normalized == target_url_normalized:
                            continue
                        
                        # Vérifier si le lien existe déjà
                        if (source_url, target_url) in existing_links_set or \
                           (source_url_normalized, target_url_normalized) in normalized_existing_links_set or \
                           (source_url_normalized, target_url_normalized) in bidirectional_links_set:
                            continue
                        
                        # Récupérer la similarité de base
                        similarity_score = similarity_matrix[source_idx][j]
                        
                        # Appliquer un bonus pour les URL prioritaires
                        if target_url in priority_urls_set:
                            similarity_score = min(similarity_score + priority_bonus, 1.0)
                        
                        # Vérifier si la similarité est suffisante
                        if similarity_score < min_similarity:
                            continue
                        
                        # Extraire des suggestions d'ancres
                        anchors = self._extract_anchor_suggestions(source_content, target_row["content1"], anchor_suggestions, target_url, gsc_queries)
                        
                        # Ajouter la suggestion obligatoire
                        mandatory_suggestions.append({
                            "target_url": target_url,
                            "similarity": similarity_score,
                            "target_type": target_type,
                            "anchors": anchors,
                            "gsc_stats": gsc_stats.get(target_url, {}),
                            "is_priority": target_url in priority_urls_set,
                            "is_mandatory": True  # Marquer comme obligatoire pour les règles de maillage
                        })
                        
                        # Mettre à jour le compteur de liens suggérés pour ce type
                        type_link_counts[target_type]["suggested"] += 1
                        
                        # Arrêter si nous avons atteint le minimum pour ce type
                        if type_link_counts[target_type]["existing"] + type_link_counts[target_type]["suggested"] >= type_link_counts[target_type]["min"]:
                            logging.debug(f"Minimum atteint pour le type {target_type} depuis {source_url}")
            
            # Deuxième passe : ajouter des suggestions facultatives basées sur la similarité
            for j in sorted_indices:
                # Ignorer la page elle-même
                if i == j:
                    continue
                
                # Récupérer les informations de la page cible
                target_row = content_df.iloc[j]
                target_url = target_row["url"]
                target_type = target_row["type"]
                
                # Ignorer la page elle-même (par URL normalisée)
                source_url_normalized = self._normalize_url(source_url)
                target_url_normalized = self._normalize_url(target_url)
                if source_url_normalized == target_url_normalized:
                    continue
                
                # Vérifier si le lien existe déjà ou est déjà dans les suggestions obligatoires
                if (source_url, target_url) in existing_links_set or \
                   (source_url_normalized, target_url_normalized) in normalized_existing_links_set or \
                   (source_url_normalized, target_url_normalized) in bidirectional_links_set or \
                   any(s["target_url"] == target_url for s in mandatory_suggestions):
                    continue
                
                # Vérifier les règles de maillage si elles sont définies
                if linking_rules and source_type in linking_rules and target_type in linking_rules[source_type]:
                    # Vérifier si nous avons déjà atteint le maximum pour ce type
                    type_counts = type_link_counts.get(target_type, {})
                    if type_counts:
                        total_links = type_counts["existing"] + type_counts["suggested"]
                        if total_links >= type_counts["max"]:
                            continue
                
                # Récupérer la similarité de base
                similarity_score = similarity_matrix[source_idx][j]
                
                # Appliquer un bonus pour les URL prioritaires
                if target_url in priority_urls_set:
                    similarity_score = min(similarity_score + priority_bonus, 1.0)
                
                # Vérifier si la similarité est suffisante
                if similarity_score < min_similarity:
                    continue
                
                # Extraire des suggestions d'ancres
                anchors = self._extract_anchor_suggestions(source_content, target_row["content1"], anchor_suggestions, target_url, gsc_queries)
                
                # Ajouter la suggestion facultative
                optional_suggestions.append({
                    "target_url": target_url,
                    "similarity": similarity_score,
                    "target_type": target_type,
                    "anchors": anchors,
                    "gsc_stats": gsc_stats.get(target_url, {}),
                    "is_priority": target_url in priority_urls_set,
                    "is_mandatory": False  # Marquer comme facultative
                })
                
                # Mettre à jour le compteur si nécessaire
                if linking_rules and source_type in linking_rules and target_type in linking_rules[source_type]:
                    type_link_counts[target_type]["suggested"] += 1
                
                # Limiter le nombre de suggestions facultatives
                remaining_slots = max_suggestions - len(mandatory_suggestions)
                if len(optional_suggestions) >= remaining_slots:
                    break
            
            # Combiner les suggestions obligatoires et facultatives
            suggestions_for_page = mandatory_suggestions + optional_suggestions
            
            # Trier par similarité décroissante
            suggestions_for_page.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Limiter au nombre maximum de suggestions
            if len(suggestions_for_page) > max_suggestions:
                suggestions_for_page = suggestions_for_page[:max_suggestions]
            
            # Ajouter les suggestions pour cette page
            for suggestion in suggestions_for_page:
                suggestions.append({
                    "source_url": source_url,
                    "source_type": source_type,
                    "target_url": suggestion["target_url"],
                    "target_type": suggestion["target_type"],
                    "similarity": suggestion["similarity"],
                    "anchor_suggestions": ", ".join(suggestion["anchors"]),
                    "clics": suggestion["gsc_stats"].get("clics", 0),
                    "impressions": suggestion["gsc_stats"].get("impressions", 0),
                    "position": suggestion["gsc_stats"].get("position", 0)
                })
        
        # Créer un DataFrame avec les suggestions
        suggestions_df = pd.DataFrame(suggestions)
        
        # Éliminer les doublons en conservant la suggestion avec la plus haute similarité
        if not suggestions_df.empty:
            # Identifier les doublons basés sur source_url et target_url
            logging.info(f"Nombre de suggestions avant déduplication: {len(suggestions_df)}")
            
            # Conserver uniquement la suggestion avec la plus haute similarité pour chaque paire source-destination
            suggestions_df = suggestions_df.sort_values(by="similarity", ascending=False)
            suggestions_df = suggestions_df.drop_duplicates(subset=["source_url", "target_url"], keep="first")
            
            logging.info(f"Nombre de suggestions après déduplication: {len(suggestions_df)}")
            
            # Trier par similarité décroissante
            suggestions_df = suggestions_df.sort_values(by="similarity", ascending=False)
        
        return suggestions_df
    
    def _extract_anchor_suggestions(self, source_text: str, target_text: str, num_suggestions: int, target_url: str, gsc_queries: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Extrait des suggestions d'ancres en priorité à partir des requêtes GSC, puis du texte cible"""
        anchors = []
        
        # 1. Essayer d'abord d'utiliser les requêtes GSC pour la page cible
        if target_url in gsc_queries and gsc_queries[target_url]:
            # Trier les requêtes par pertinence (clics, puis impressions si égalité)
            sorted_queries = sorted(
                gsc_queries[target_url],
                key=lambda q: (q["clics"], q["impressions"]),
                reverse=True
            )
            
            # Filtrer les requêtes pour ne garder que celles qui sont pertinentes
            filtered_queries = []
            for q in sorted_queries:
                query = q["query"]
                # Vérifier que la requête est assez longue mais pas trop
                words = query.split()
                if 1 < len(words) < 8 and len(query) > 3:
                    # Vérifier que la requête est grammaticalement correcte (heuristique simple)
                    if self._is_grammatically_valid_french(query):
                        filtered_queries.append(query)
            
            # Ajouter les meilleures requêtes comme suggestions d'ancres
            anchors.extend(filtered_queries[:num_suggestions])
        
        # 2. Si on n'a pas assez de requêtes GSC, compléter avec des extraits du contenu
        if len(anchors) < num_suggestions and isinstance(target_text, str):
            # Extraire les phrases du texte cible
            sentences = target_text.split('.')
            
            # Filtrer les phrases trop courtes ou trop longues
            sentences = [s.strip() for s in sentences if 10 < len(s.strip()) < 100]
            
            # Ajouter les phrases les plus pertinentes
            for s in sentences:
                if len(anchors) >= num_suggestions:
                    break
                    
                # Limiter à 5-7 mots
                words = s.split()
                if len(words) > 7:
                    s = ' '.join(words[:7]) + '...'
                
                # Éviter les doublons
                if s not in anchors:
                    anchors.append(s)
        
        # 3. Si on n'a toujours pas assez de suggestions, utiliser des mots-clés
        if len(anchors) < num_suggestions and isinstance(target_text, str):
            # Extraire les mots-clés (mots les plus fréquents qui ne sont pas des stopwords)
            words = [w for w in target_text.lower().split() if w not in STOP_WORDS and len(w) > 3]
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Trier par fréquence
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Ajouter les mots-clés les plus fréquents
            for word, _ in keywords:
                if len(anchors) >= num_suggestions:
                    break
                if word not in anchors:
                    anchors.append(word)
        
        return anchors[:num_suggestions]
    
    def _is_grammatically_valid_french(self, text: str) -> bool:
        """Vérifie si une phrase est grammaticalement valide en français (heuristique simple)"""
        # Liste de prépositions et articles français courants
        french_prepositions = ['à', 'au', 'aux', 'de', 'des', 'du', 'en', 'par', 'pour', 'sur', 'avec', 'sans']
        french_articles = ['le', 'la', 'les', 'un', 'une', 'des', 'l\'', 'l']
        
        # Vérifier si le texte est vide
        if not text or not isinstance(text, str):
            return False
            
        words = text.lower().split()
        
        # Texte trop court (un seul mot)
        if len(words) < 2:
            return False
            
        # Vérifier les cas où des prépositions sont nécessaires
        for i, word in enumerate(words[:-1]):
            # Cas typiques où une préposition est attendue entre deux noms
            if i > 0 and word not in french_prepositions and word not in french_articles:
                next_word = words[i+1]
                prev_word = words[i-1]
                
                # Cas spécifiques où une préposition manque probablement
                if all(w not in french_prepositions and w not in french_articles for w in [prev_word, word, next_word]):
                    # Trois noms consécutifs sans préposition ni article
                    if i > 1 and words[i-2] not in french_prepositions and words[i-2] not in french_articles:
                        return False
        
        # Si on arrive ici, la phrase semble correcte
        return True
    
    def _save_results(self, suggestions_df: pd.DataFrame) -> str:
        """Sauvegarde les résultats dans un fichier Excel"""
        # Créer un nom de fichier unique basé sur la date et l'heure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = "app/results"
        os.makedirs(result_dir, exist_ok=True)
        result_file = f"{result_dir}/maillage_{timestamp}.xlsx"
        
        # Sauvegarder en Excel
        with pd.ExcelWriter(result_file, engine='openpyxl') as writer:
            suggestions_df.to_excel(writer, sheet_name='Suggestions', index=False)
        
        return result_file
