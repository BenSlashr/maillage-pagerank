import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os
import json
import time
import asyncio
from datetime import datetime
import torch
from typing import Dict, List, Callable, Optional, Tuple, Any
import nltk
from nltk.corpus import stopwords
from urllib.parse import urlparse

# Télécharger les stopwords NLTK si nécessaire
try:
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = set(stopwords.words('french'))
except:
    STOP_WORDS = set()

# Configuration du modèle BERT
MODEL_NAME = "distiluse-base-multilingual-cased-v2"
BATCH_SIZE = 32

class SEOAnalyzer:
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
            content_file: Chemin vers le fichier de contenu
            links_file: Chemin vers le fichier de liens existants
            gsc_file: Chemin vers le fichier GSC
            min_similarity: Score minimum de similarité pour les suggestions
            anchor_suggestions: Nombre de suggestions d'ancres
            linking_rules: Règles de maillage entre segments
            priority_urls: Liste des URL prioritaires dont le PageRank doit être maintenu ou amélioré
            priority_urls_strict: Si True, les URL prioritaires doivent améliorer leur PageRank; sinon, elles doivent au moins le maintenir
            
        Returns:
            Chemin vers le fichier de résultats
        """
        start_time = time.time()
        logging.info("Début de l'analyse SEO")
        
        # Charger les fichiers
        if self.progress_callback:
            await self.progress_callback("Initialisation de l'analyse...", 0, 5)
        
        # Charger le fichier de contenu
        content_df = self._load_content_file(content_file)
        num_pages_original = len(content_df)
        if self.progress_callback:
            await self.progress_callback(f"Fichier de contenu chargé ({num_pages_original} pages)", 1, 5)
        
        # Charger le fichier de liens existants si fourni
        existing_links = self._load_links_file(links_file) if links_file else pd.DataFrame(columns=["Source", "Destination"])
        num_existing_links = len(existing_links)
        if self.progress_callback:
            await self.progress_callback(f"Fichier de liens existants chargé ({num_existing_links} liens)", 2, 5)
        
        # Charger le fichier GSC si fourni
        gsc_data = self._load_gsc_file(gsc_file) if gsc_file else pd.DataFrame(columns=["URL", "Clics", "Impressions", "Position"])
        num_gsc_entries = len(gsc_data)
        if self.progress_callback:
            await self.progress_callback(f"Fichier GSC chargé ({num_gsc_entries} entrées)", 3, 5)
        
        # Prétraiter les données
        if self.progress_callback:
            await self.progress_callback("Prétraitement des données...", 4, 5)
        content_df = self._preprocess_content(content_df)
        num_pages_after_preprocess = len(content_df)
        
        # Journaliser si des pages ont été filtrées pendant le prétraitement
        if num_pages_original != num_pages_after_preprocess:
            logging.info(f"{num_pages_original - num_pages_after_preprocess} pages ont été filtrées pendant le prétraitement")
            if self.progress_callback:
                await self.progress_callback(f"Prétraitement terminé - {num_pages_after_preprocess}/{num_pages_original} pages conservées", 5, 5)
        else:
            if self.progress_callback:
                await self.progress_callback(f"Prétraitement terminé pour {num_pages_after_preprocess} pages", 5, 5)
        
        # Générer les embeddings pour le contenu
        if self.progress_callback:
            await self.progress_callback(f"Préparation des embeddings pour {num_pages_after_preprocess} pages...", 0, content_df.shape[0])
        embeddings = self._generate_embeddings(content_df)
        
        # Calculer la matrice de similarité
        if self.progress_callback:
            await self.progress_callback("Calcul de la matrice de similarité...", 0, 1)
        similarity_matrix = cosine_similarity(embeddings)
        if self.progress_callback:
            await self.progress_callback(f"Matrice de similarité calculée ({num_pages_after_preprocess}x{num_pages_after_preprocess})", 1, 1)
        
        # Vérifier que la taille de la matrice de similarité correspond au nombre de lignes dans content_df
        if similarity_matrix.shape[0] != content_df.shape[0]:
            error_msg = f"Incohérence de dimensions: matrice de similarité ({similarity_matrix.shape[0]}x{similarity_matrix.shape[1]}) vs DataFrame de contenu ({content_df.shape[0]} lignes)"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        # Générer les suggestions de liens
        if self.progress_callback:
            await self.progress_callback(f"Début de l'analyse des {num_pages_after_preprocess} pages...", 0, num_pages_after_preprocess)
            
        # Journaliser les URL prioritaires si définies
        if priority_urls and len(priority_urls) > 0:
            logging.info(f"Analyse avec {len(priority_urls)} URL prioritaires définies")
            logging.info(f"Mode strict pour les URL prioritaires: {priority_urls_strict}")
            
        suggestions_df = self._generate_suggestions(
            content_df,
            similarity_matrix,
            existing_links,
            gsc_data,
            min_similarity,
            anchor_suggestions,
            linking_rules,
            priority_urls,
            priority_urls_strict
        )
        
        # Sauvegarder les résultats
        if self.progress_callback:
            await self.progress_callback("Sauvegarde des résultats...", 0, 1)
        result_file = self._save_results(suggestions_df)
        if self.progress_callback:
            await self.progress_callback("Analyse terminée", 1, 1)
        
        elapsed_time = time.time() - start_time
        logging.info(f"Analyse terminée en {elapsed_time:.2f} secondes")
        
        return result_file
    
    def _load_content_file(self, file_path: str) -> pd.DataFrame:
        """Charge le fichier de contenu"""
        try:
            df = pd.read_excel(file_path)
            required_columns = ["Adresse", "Segments", "Extracteur 1 1"]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Colonne requise manquante dans le fichier de contenu: {col}")
            
            # Renommer les colonnes pour faciliter l'accès
            column_mapping = {
                "Adresse": "url",
                "Segments": "type",
                "Extracteur 1 1": "content1",
                "Extracteur 2 1": "content2" if "Extracteur 2 1" in df.columns else None
            }
            
            # Supprimer les mappings None
            column_mapping = {k: v for k, v in column_mapping.items() if v is not None}
            
            df = df.rename(columns=column_mapping)
            
            # Filtrer les lignes avec du contenu
            df = df.dropna(subset=["content1"])
            
            return df
        except Exception as e:
            logging.error(f"Erreur lors du chargement du fichier de contenu: {str(e)}")
            raise e
    
    def _load_links_file(self, file_path: str) -> pd.DataFrame:
        """Charge le fichier de liens existants"""
        try:
            df = pd.read_excel(file_path)
            required_columns = ["Source", "Destination"]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Colonne requise manquante dans le fichier de liens: {col}")
            
            return df
        except Exception as e:
            logging.error(f"Erreur lors du chargement du fichier de liens: {str(e)}")
            raise e
    
    def _load_gsc_file(self, file_path: str) -> pd.DataFrame:
        """Charge le fichier GSC"""
        try:
            df = pd.read_excel(file_path)
            
            # Valider les colonnes requises selon app/main.py
            required_columns = ["Query", "Page", "Clicks", "Impressions", "CTR", "Position"]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Colonne requise manquante dans le fichier GSC: {col}")
            
            # Renommer les colonnes pour correspondre à ce qui est attendu dans le reste du code
            column_mapping = {
                "Page": "URL",
                "Clicks": "Clics",
                "Impressions": "Impressions",
                "Position": "Position"
            }
            
            # Appliquer le renommage uniquement pour les colonnes qui existent
            rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
            if rename_dict:
                df = df.rename(columns=rename_dict)
            
            logging.info(f"Fichier GSC chargé avec succès: {len(df)} lignes")
            return df
        except Exception as e:
            logging.error(f"Erreur lors du chargement du fichier GSC: {str(e)}")
            raise e
    
    def _preprocess_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prétraite les données de contenu"""
        # Supprimer les lignes avec des URLs invalides
        df = df[df["url"].apply(lambda x: isinstance(x, str) and x.startswith(("http://", "https://")))]
        
        # Normaliser les types de page
        df["type"] = df["type"].str.lower().str.strip()
        df["type"] = df["type"].apply(
            lambda x: 'blog' if isinstance(x, str) and ('blog' in x or 'article' in x) 
            else 'categorie' if isinstance(x, str) and 'categ' in x 
            else 'produit' if isinstance(x, str) and ('produit' in x or 'product' in x) 
            else x
        )
        
        # Extraire les domaines des URLs
        df["domain"] = df["url"].apply(lambda x: urlparse(x).netloc if isinstance(x, str) else "")
        
        # Nettoyer le contenu
        df["clean_content"] = df["content1"].apply(self._clean_text)
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """Nettoie un texte en supprimant les stopwords et les caractères spéciaux"""
        if not isinstance(text, str):
            return ""
        
        # Convertir en minuscules
        text = text.lower()
        
        # Supprimer les caractères spéciaux
        text = ''.join([c if c.isalnum() or c.isspace() else ' ' for c in text])
        
        # Supprimer les stopwords
        if STOP_WORDS:
            words = text.split()
            words = [w for w in words if w not in STOP_WORDS]
            text = ' '.join(words)
        
        return text
    
    def _generate_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """Génère les embeddings pour le contenu"""
        texts = df["clean_content"].tolist()
        
        # Générer les embeddings par lots pour économiser la mémoire
        embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            batch_embeddings = self.model.encode(batch_texts)
            embeddings.append(batch_embeddings)
            
            # Mettre à jour la progression
            if self.progress_callback:
                asyncio.create_task(self.progress_callback(f"Génération des embeddings ({i+len(batch_texts)}/{len(texts)})", i+len(batch_texts), len(texts)))
        
        return np.vstack(embeddings)
    
    def _generate_suggestions(
        self,
        content_df: pd.DataFrame,
        similarity_matrix: np.ndarray,
        existing_links: pd.DataFrame,
        gsc_data: pd.DataFrame,
        min_similarity: float,
        anchor_suggestions: int,
        linking_rules: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None,
        priority_urls: Optional[List[str]] = None,
        priority_urls_strict: bool = False
    ) -> pd.DataFrame:
        """
        Génère des suggestions de liens internes en tenant compte des URL prioritaires.
        
        Args:
            content_df: DataFrame contenant le contenu des pages
            similarity_matrix: Matrice de similarité entre les pages
            existing_links: DataFrame des liens existants
            gsc_data: DataFrame des données GSC
            min_similarity: Score minimum de similarité pour les suggestions
            anchor_suggestions: Nombre de suggestions d'ancres
            linking_rules: Règles de maillage entre segments
            priority_urls: Liste des URL prioritaires dont le PageRank doit être maintenu ou amélioré
            priority_urls_strict: Si True, les URL prioritaires doivent améliorer leur PageRank; sinon, elles doivent au moins le maintenir
        
        Returns:
            DataFrame des suggestions de liens
        """
        # Créer un dictionnaire pour stocker les suggestions
        suggestions = []
        
        # Convertir les URL prioritaires en ensemble pour une recherche plus rapide
        priority_urls_set = set(priority_urls) if priority_urls else set()
        
        # Créer un ensemble des liens existants pour une recherche plus rapide
        existing_links_set = set()
        for _, row in existing_links.iterrows():
            source = row["Source"]
            destination = row["Destination"]
            if isinstance(source, str) and isinstance(destination, str):
                existing_links_set.add((source, destination))
        
        # Créer un dictionnaire pour les statistiques GSC et les requêtes
        gsc_stats = {}
        gsc_queries = {}
        if not gsc_data.empty:
            for _, row in gsc_data.iterrows():
                url = row["URL"]
                if isinstance(url, str):
                    # Stocker les statistiques
                    if url not in gsc_stats:
                        gsc_stats[url] = {
                            "clics": row["Clics"],
                            "impressions": row["Impressions"],
                            "position": row["Position"]
                        }
                    
                    # Stocker les requêtes pour chaque URL
                    if url not in gsc_queries:
                        gsc_queries[url] = []
                    
                    # Ajouter la requête avec ses métriques pour le tri ultérieur
                    if "Query" in row and isinstance(row["Query"], str) and len(row["Query"]) > 0:
                        gsc_queries[url].append({
                            "query": row["Query"],
                            "clics": row["Clics"] if isinstance(row["Clics"], (int, float)) else 0,
                            "impressions": row["Impressions"] if isinstance(row["Impressions"], (int, float)) else 0,
                            "position": row["Position"] if isinstance(row["Position"], (int, float)) else 0
                        })
        
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
                
                # Récupérer la similarité de base
                similarity_score = similarities[j]
                
                # Récupérer les informations de la page cible
                target_row = content_df.iloc[j]
                target_url = target_row["url"]
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
        # Générer un nom de fichier unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"app/results/maillage_{timestamp}.xlsx"
        
        # Sauvegarder en Excel
        with pd.ExcelWriter(result_file, engine='openpyxl') as writer:
            suggestions_df.to_excel(writer, sheet_name='Suggestions', index=False)
        
        return result_file
