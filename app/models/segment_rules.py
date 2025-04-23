"""
Module pour la gestion des règles de segmentation des pages.
Permet de définir des règles basées sur des patterns d'URL pour attribuer
des segments (types) aux pages lors du crawl.
"""

import re
import os
import json
import logging
from typing import Dict, List, Optional, Pattern, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class SegmentRule:
    """
    Règle de segmentation pour les pages.
    Une règle peut être basée sur un pattern d'URL (regex) ou un chemin d'URL.
    """
    
    def __init__(
        self,
        pattern: str,
        segment: str,
        is_regex: bool = False,
        priority: int = 0,
        description: str = ""
    ):
        """
        Initialise une règle de segmentation.
        
        Args:
            pattern: Pattern d'URL (regex ou chemin)
            segment: Segment (type) à attribuer aux pages correspondantes
            is_regex: Si True, le pattern est une expression régulière
            priority: Priorité de la règle (les règles de priorité plus élevée sont appliquées en premier)
            description: Description optionnelle de la règle
        """
        self.pattern = pattern
        self.segment = segment
        self.is_regex = is_regex
        self.priority = priority
        self.description = description
        self._compiled_regex: Optional[Pattern] = None
        
        # Compiler l'expression régulière si nécessaire
        if is_regex:
            try:
                self._compiled_regex = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                logger.error(f"Erreur lors de la compilation de l'expression régulière '{pattern}': {str(e)}")
                raise ValueError(f"Expression régulière invalide: {str(e)}")
    
    def matches(self, url: str) -> bool:
        """
        Vérifie si l'URL correspond à cette règle.
        
        Args:
            url: URL à vérifier
            
        Returns:
            True si l'URL correspond à la règle, False sinon
        """
        parsed_url = urlparse(url)
        path = parsed_url.path
        
        if self.is_regex and self._compiled_regex:
            # Pour les regex, on vérifie l'URL complète
            return bool(self._compiled_regex.search(url))
        else:
            # Pour les chemins simples, on vérifie si le chemin commence par le pattern
            # ou si le pattern est un segment du chemin
            path_segments = [seg for seg in path.split('/') if seg]
            
            # Vérifier si le chemin commence par le pattern
            if path.startswith(self.pattern):
                return True
            
            # Vérifier si le pattern est un segment du chemin
            for segment in path_segments:
                if segment == self.pattern:
                    return True
            
            return False
    
    def to_dict(self) -> Dict[str, Union[str, bool, int]]:
        """
        Convertit la règle en dictionnaire pour la sérialisation.
        
        Returns:
            Dictionnaire représentant la règle
        """
        return {
            "pattern": self.pattern,
            "segment": self.segment,
            "is_regex": self.is_regex,
            "priority": self.priority,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Union[str, bool, int]]) -> 'SegmentRule':
        """
        Crée une règle à partir d'un dictionnaire.
        
        Args:
            data: Dictionnaire représentant la règle
            
        Returns:
            Instance de SegmentRule
        """
        return cls(
            pattern=str(data.get("pattern", "")),
            segment=str(data.get("segment", "")),
            is_regex=bool(data.get("is_regex", False)),
            priority=int(data.get("priority", 0)),
            description=str(data.get("description", ""))
        )


class SegmentRuleManager:
    """
    Gestionnaire des règles de segmentation.
    Permet de définir, charger, sauvegarder et appliquer des règles de segmentation.
    """
    
    def __init__(self, rules_file: str = "app/data/segment_rules.json"):
        """
        Initialise le gestionnaire de règles.
        
        Args:
            rules_file: Chemin vers le fichier de règles
        """
        self.rules_file = rules_file
        self.rules: List[SegmentRule] = []
        self.default_segment = "page"  # Segment par défaut si aucune règle ne correspond
        
        # Charger les règles si le fichier existe
        if os.path.exists(rules_file):
            try:
                self.load_rules()
            except Exception as e:
                logger.error(f"Erreur lors du chargement des règles: {str(e)}")
                # Initialiser avec des règles par défaut
                self._init_default_rules()
        else:
            # Initialiser avec des règles par défaut
            self._init_default_rules()
    
    def _init_default_rules(self):
        """Initialise des règles par défaut."""
        self.rules = [
            SegmentRule(
                pattern="blog",
                segment="blog",
                is_regex=False,
                priority=10,
                description="Pages de blog"
            ),
            SegmentRule(
                pattern="produit",
                segment="produit",
                is_regex=False,
                priority=10,
                description="Pages de produit"
            ),
            SegmentRule(
                pattern="categorie",
                segment="categorie",
                is_regex=False,
                priority=10,
                description="Pages de catégorie"
            ),
            SegmentRule(
                pattern=r"/article/.*",
                segment="article",
                is_regex=True,
                priority=20,
                description="Articles (regex)"
            ),
            SegmentRule(
                pattern=r"/actualites/.*",
                segment="actualite",
                is_regex=True,
                priority=20,
                description="Actualités (regex)"
            )
        ]
    
    def add_rule(self, rule: SegmentRule) -> None:
        """
        Ajoute une règle au gestionnaire.
        
        Args:
            rule: Règle à ajouter
        """
        self.rules.append(rule)
        # Trier les règles par priorité (décroissante)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def remove_rule(self, index: int) -> None:
        """
        Supprime une règle du gestionnaire.
        
        Args:
            index: Index de la règle à supprimer
        """
        if 0 <= index < len(self.rules):
            del self.rules[index]
    
    def get_segment(self, url: str) -> str:
        """
        Détermine le segment d'une URL en appliquant les règles.
        
        Args:
            url: URL à analyser
            
        Returns:
            Segment (type) de la page
        """
        for rule in self.rules:
            if rule.matches(url):
                logger.debug(f"URL {url} correspond à la règle {rule.pattern} -> segment {rule.segment}")
                return rule.segment
        
        # Si aucune règle ne correspond, utiliser le segment par défaut
        return self.default_segment
    
    def load_rules(self) -> None:
        """
        Charge les règles depuis le fichier.
        """
        try:
            with open(self.rules_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Charger les règles
                rules_data = data.get("rules", [])
                self.rules = [SegmentRule.from_dict(rule_data) for rule_data in rules_data]
                
                # Charger le segment par défaut
                self.default_segment = data.get("default_segment", "page")
                
                # Trier les règles par priorité
                self.rules.sort(key=lambda r: r.priority, reverse=True)
                
                logger.info(f"Règles chargées depuis {self.rules_file}: {len(self.rules)} règles")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des règles: {str(e)}")
            raise
    
    def save_rules(self) -> None:
        """
        Sauvegarde les règles dans le fichier.
        """
        try:
            # Créer le répertoire parent si nécessaire
            os.makedirs(os.path.dirname(self.rules_file), exist_ok=True)
            
            data = {
                "default_segment": self.default_segment,
                "rules": [rule.to_dict() for rule in self.rules]
            }
            
            with open(self.rules_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Règles sauvegardées dans {self.rules_file}: {len(self.rules)} règles")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des règles: {str(e)}")
            raise
    
    def import_rules(self, rules_data: Dict) -> None:
        """
        Importe des règles depuis un dictionnaire.
        
        Args:
            rules_data: Dictionnaire contenant les règles
        """
        try:
            # Charger le segment par défaut
            self.default_segment = rules_data.get("default_segment", "page")
            
            # Charger les règles
            rules_list = rules_data.get("rules", [])
            self.rules = [SegmentRule.from_dict(rule_data) for rule_data in rules_list]
            
            # Trier les règles par priorité
            self.rules.sort(key=lambda r: r.priority, reverse=True)
            
            logger.info(f"Règles importées: {len(self.rules)} règles")
        except Exception as e:
            logger.error(f"Erreur lors de l'importation des règles: {str(e)}")
            raise
    
    def export_rules(self) -> Dict:
        """
        Exporte les règles sous forme de dictionnaire.
        
        Returns:
            Dictionnaire contenant les règles
        """
        return {
            "default_segment": self.default_segment,
            "rules": [rule.to_dict() for rule in self.rules]
        }
