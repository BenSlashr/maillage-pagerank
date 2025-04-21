"""
Module pour la mise en cache des calculs de PageRank.
"""
import logging
import time
from typing import Dict, Any, Optional, Tuple

class PageRankCache:
    """
    Classe pour mettre en cache les résultats des calculs de PageRank.
    Cela permet d'éviter les calculs redondants et d'améliorer les performances.
    """
    def __init__(self, cache_duration_seconds: int = 300):
        """
        Initialise le cache de PageRank.
        
        Args:
            cache_duration_seconds: Durée de validité du cache en secondes (par défaut: 5 minutes)
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = cache_duration_seconds
        logging.info(f"Cache PageRank initialisé avec une durée de validité de {cache_duration_seconds} secondes")
    
    def _generate_cache_key(self, job_id: str, content_links_only: bool, alpha: float, beta: float) -> str:
        """
        Génère une clé de cache unique basée sur les paramètres de calcul.
        
        Args:
            job_id: Identifiant de la tâche
            content_links_only: Si True, ne prend en compte que les liens dans le contenu principal
            alpha: Coefficient pour la pondération sémantique
            beta: Coefficient pour la pondération par position
            
        Returns:
            Clé de cache unique
        """
        return f"{job_id}_{content_links_only}_{alpha}_{beta}"
    
    def get(self, job_id: str, content_links_only: bool, alpha: float, beta: float) -> Optional[Dict[str, Any]]:
        """
        Récupère les résultats du cache si disponibles et valides.
        
        Args:
            job_id: Identifiant de la tâche
            content_links_only: Si True, ne prend en compte que les liens dans le contenu principal
            alpha: Coefficient pour la pondération sémantique
            beta: Coefficient pour la pondération par position
            
        Returns:
            Résultats mis en cache ou None si non disponibles
        """
        cache_key = self._generate_cache_key(job_id, content_links_only, alpha, beta)
        
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            current_time = time.time()
            
            # Vérifier si le cache est encore valide
            if current_time - cache_entry["timestamp"] < self.cache_duration:
                logging.info(f"Utilisation du cache PageRank pour {cache_key}")
                return cache_entry["data"]
            else:
                logging.info(f"Cache PageRank expiré pour {cache_key}")
                # Supprimer l'entrée expirée
                del self.cache[cache_key]
        
        return None
    
    def set(self, job_id: str, content_links_only: bool, alpha: float, beta: float, data: Dict[str, Any]) -> None:
        """
        Stocke les résultats dans le cache.
        
        Args:
            job_id: Identifiant de la tâche
            content_links_only: Si True, ne prend en compte que les liens dans le contenu principal
            alpha: Coefficient pour la pondération sémantique
            beta: Coefficient pour la pondération par position
            data: Données à mettre en cache
        """
        cache_key = self._generate_cache_key(job_id, content_links_only, alpha, beta)
        
        self.cache[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }
        
        logging.info(f"Mise en cache des résultats PageRank pour {cache_key}")
    
    def invalidate(self, job_id: str) -> None:
        """
        Invalide toutes les entrées de cache pour un job_id spécifique.
        À utiliser lorsque les données sous-jacentes changent.
        
        Args:
            job_id: Identifiant de la tâche
        """
        keys_to_remove = [key for key in self.cache if key.startswith(f"{job_id}_")]
        
        for key in keys_to_remove:
            del self.cache[key]
        
        if keys_to_remove:
            logging.info(f"Cache PageRank invalidé pour le job {job_id} ({len(keys_to_remove)} entrées supprimées)")
    
    def clear(self) -> None:
        """
        Vide complètement le cache.
        """
        self.cache.clear()
        logging.info("Cache PageRank entièrement vidé")
