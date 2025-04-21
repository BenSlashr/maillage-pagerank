"""
Module pour la gestion des URL prioritaires dont le PageRank doit être maintenu ou amélioré.
"""
import os
import json
import logging
from typing import List, Dict, Optional, Set, Tuple
import pandas as pd

class PriorityURLManager:
    """
    Gestionnaire des URL prioritaires.
    """
    def __init__(self, job_id: str):
        """
        Initialise le gestionnaire d'URL prioritaires.
        
        Args:
            job_id: Identifiant de la tâche d'analyse
        """
        self.job_id = job_id
        self.priority_urls: Set[str] = set()
        self.base_dir = os.path.join("app", "data", "jobs", job_id)
        self.priority_file = os.path.join(self.base_dir, "priority_urls.json")
        self._load_priority_urls()
    
    def _load_priority_urls(self) -> None:
        """
        Charge les URL prioritaires depuis le fichier JSON.
        """
        if os.path.exists(self.priority_file):
            try:
                with open(self.priority_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.priority_urls = set(data.get("urls", []))
                logging.info(f"Chargement de {len(self.priority_urls)} URL prioritaires")
            except Exception as e:
                logging.error(f"Erreur lors du chargement des URL prioritaires: {str(e)}")
                self.priority_urls = set()
    
    def _save_priority_urls(self) -> None:
        """
        Sauvegarde les URL prioritaires dans un fichier JSON.
        """
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            with open(self.priority_file, 'w', encoding='utf-8') as f:
                json.dump({"urls": list(self.priority_urls)}, f, ensure_ascii=False, indent=2)
            logging.info(f"Sauvegarde de {len(self.priority_urls)} URL prioritaires")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde des URL prioritaires: {str(e)}")
    
    def add_priority_urls(self, urls: List[str]) -> None:
        """
        Ajoute des URL à la liste des URL prioritaires.
        
        Args:
            urls: Liste des URL à ajouter
        """
        for url in urls:
            self.priority_urls.add(url.strip())
        self._save_priority_urls()
    
    def remove_priority_urls(self, urls: List[str]) -> None:
        """
        Supprime des URL de la liste des URL prioritaires.
        
        Args:
            urls: Liste des URL à supprimer
        """
        for url in urls:
            if url.strip() in self.priority_urls:
                self.priority_urls.remove(url.strip())
        self._save_priority_urls()
    
    def get_priority_urls(self) -> List[str]:
        """
        Retourne la liste des URL prioritaires.
        
        Returns:
            Liste des URL prioritaires
        """
        return list(self.priority_urls)
    
    def clear_priority_urls(self) -> None:
        """
        Vide la liste des URL prioritaires.
        """
        self.priority_urls.clear()
        self._save_priority_urls()
    
    def analyze_pagerank_impact(self, current_pagerank: Dict[str, float], 
                               optimized_pagerank: Dict[str, float]) -> Dict[str, Dict]:
        """
        Analyse l'impact du plan de maillage sur les URL prioritaires.
        
        Args:
            current_pagerank: Dictionnaire des scores PageRank actuels
            optimized_pagerank: Dictionnaire des scores PageRank optimisés
            
        Returns:
            Dictionnaire avec l'analyse d'impact pour chaque URL prioritaire
        """
        impact_analysis = {}
        
        for url in self.priority_urls:
            if url in current_pagerank and url in optimized_pagerank:
                current_score = current_pagerank[url]
                optimized_score = optimized_pagerank[url]
                
                # Calculer le pourcentage d'amélioration
                improvement_pct = ((optimized_score - current_score) / current_score * 100) if current_score > 0 else 0
                
                # Déterminer le statut (amélioré, maintenu, dégradé)
                if improvement_pct > 1:  # Plus de 1% d'amélioration
                    status = "improved"
                elif improvement_pct >= -1:  # Entre -1% et 1% (considéré comme maintenu)
                    status = "maintained"
                else:  # Moins de -1% (dégradé)
                    status = "degraded"
                
                impact_analysis[url] = {
                    "current_pagerank": current_score,
                    "optimized_pagerank": optimized_score,
                    "improvement_percentage": round(improvement_pct, 2),
                    "status": status
                }
            else:
                # URL non trouvée dans les données de PageRank
                impact_analysis[url] = {
                    "current_pagerank": None,
                    "optimized_pagerank": None,
                    "improvement_percentage": None,
                    "status": "not_found"
                }
        
        return impact_analysis
