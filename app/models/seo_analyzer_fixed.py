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
        
        # Charger le contenu des pages
        content_df = pd.read_excel(content_file)
        
        # Vérifier les colonnes requises
        required_columns = ["url", "title", "content1", "type"]
        for col in required_columns:
            if col not in content_df.columns:
                raise ValueError(f"Colonne requise manquante dans le fichier de contenu: {col}")
        
        # Charger les liens existants si disponibles
        existing_links_df = None
        if links_file and os.path.exists(links_file):
            existing_links_df = pd.read_excel(links_file)
            
            # Vérifier les colonnes requises
            required_columns = ["source", "target"]
            for col in required_columns:
                if col not in existing_links_df.columns:
                    raise ValueError(f"Colonne requise manquante dans le fichier de liens: {col}")
        
        # Charger les données GSC si disponibles
        gsc_data = None
        if gsc_file and os.path.exists(gsc_file):
            gsc_data = pd.read_excel(gsc_file)
            
            # Vérifier les colonnes requises
            required_columns = ["page", "query", "clicks", "impressions", "position"]
            for col in required_columns:
                if col not in gsc_data.columns:
                    raise ValueError(f"Colonne requise manquante dans le fichier GSC: {col}")
        
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
        # Supprimer les paramètres d'URL
        url = url.split('?')[0].split('#')[0]
        
        # Supprimer le slash final s'il existe
        if url.endswith('/'):
            url = url[:-1]
            
        # Convertir en minuscules
        url = url.lower()
        
        return url
    
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
        if existing_links_df is not None:
            for _, row in existing_links_df.iterrows():
                existing_links_set.add((row["source"], row["target"]))
        
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
        
        # Parcourir chaque page pour générer des suggestions
        for i, source_row in enumerate(content_df.itertuples()):
            source_url = source_row.url
            source_type = source_row.type
            source_content = source_row.content1
            
            # Mettre à jour la progression
            if self.progress_callback:
                asyncio.create_task(self.progress_callback(f"Analyse de la page {i+1}/{len(content_df)}: {source_url}", i+1, len(content_df)))
            
            # Obtenir les scores de similarité pour cette page
            similarities = similarity_matrix[i]
            
            # Trier les indices par score de similarité (du plus élevé au plus bas)
            sorted_indices = np.argsort(similarities)[::-1]
            
            # Filtrer les pages avec une similarité suffisante
            suggestions_for_page = []
            
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
            
            for j in sorted_indices:
                # Ignorer la page elle-même
                if i == j:
                    continue
                
                # Récupérer les informations de la page cible
                target_row = content_df.iloc[j]
                target_url = target_row["url"]
                
                # Ignorer la page elle-même (par URL normalisée)
                # Normaliser les URLs pour gérer les cas avec/sans slash final ou paramètres
                source_url_normalized = self._normalize_url(source_url)
                target_url_normalized = self._normalize_url(target_url)
                if source_url_normalized == target_url_normalized:
                    logging.debug(f"Ignoré auto-lien: {source_url} -> {target_url}")
                    continue
                
                # Récupérer la similarité de base
                similarity_score = similarities[j]
                target_type = target_row["type"]
                
                if target_url in priority_urls_set:
                    # Appliquer un bonus pour favoriser les liens vers les URL prioritaires
                    similarity_score += priority_bonus
                    logging.debug(f"Bonus de similarité appliqué pour l'URL prioritaire: {target_url}")
                
                # Vérifier si la similarité (avec ou sans bonus) est suffisante
                if similarity_score < min_similarity:
                    continue
                
                # Vérifier si le lien existe déjà
                if (source_url, target_url) in existing_links_set:
                    continue
                
                # Vérifier les règles de maillage si elles sont définies
                if linking_rules and source_type in linking_rules and target_type in linking_rules[source_type]:
                    rule = linking_rules[source_type][target_type]
                    min_links = rule.get("min_links", 0)
                    max_links = rule.get("max_links", 10)
                    
                    # Compter le nombre de liens existants de ce type
                    existing_count = sum(1 for s, d in existing_links_set if s == source_url and content_df[content_df["url"] == d]["type"].iloc[0] == target_type)
                    
                    # Vérifier si on a déjà atteint le nombre maximum de liens
                    if existing_count >= max_links:
                        continue
                
                # Extraire des suggestions d'ancres basées sur les requêtes GSC
                anchors = self._extract_anchor_suggestions(source_content, target_row["content1"], anchor_suggestions, target_url, gsc_queries)
                
                # Ajouter la suggestion
                suggestions_for_page.append({
                    "target_url": target_url,
                    "similarity": similarity_score,  # Utiliser le score avec bonus si appliqué
                    "target_type": target_type,
                    "anchors": anchors,
                    "gsc_stats": gsc_stats.get(target_url, {}),
                    "is_priority": target_url in priority_urls_set  # Marquer si c'est une URL prioritaire
                })
                
                # Limiter le nombre de suggestions selon max_suggestions
                if len(suggestions_for_page) >= max_suggestions:
                    break
            
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
        
        # Trier par similarité décroissante
        if not suggestions_df.empty:
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
