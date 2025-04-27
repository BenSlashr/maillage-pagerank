"""
Module de crawling pour l'application de maillage interne.
Ce module permet de crawler un site web et d'extraire le contenu et les liens internes
pour les utiliser dans l'analyse de maillage.
"""

import os
import re
import json
import logging
import asyncio
import time
from datetime import datetime
from urllib.parse import urlparse, urljoin, urldefrag
from typing import Dict, List, Set, Optional, Callable, Tuple, Any

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Import du gestionnaire de règles de segmentation
from app.models.segment_rules import SegmentRuleManager

# Configuration du logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class WebCrawler:
    """
    Classe pour crawler un site web et extraire son contenu et ses liens internes.
    Utilise requests et BeautifulSoup pour le crawling et l'extraction de contenu.
    """
    
    def __init__(
        self,
        start_url: str,
        max_pages: int = 100,
        max_depth: int = 5,
        respect_robots: bool = True,
        crawl_delay: float = 0.5,
        timeout: float = 10.0,
        max_retries: int = 3,
        user_agent: str = "MaillageBot/1.0",
        segment_rules_file: Optional[str] = None,
        exclude_from_linking_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ):
        """
        Initialise le crawler.
        
        Args:
            start_url: URL de départ pour le crawl
            max_pages: Nombre maximum de pages à crawler
            respect_robots: Si True, respecte les règles du robots.txt
            crawl_delay: Délai entre les requêtes en secondes
            timeout: Timeout des requêtes HTTP en secondes
            max_retries: Nombre maximum de tentatives pour une URL
            user_agent: User-Agent à utiliser pour les requêtes
            segment_rules_file: Chemin vers le fichier de règles de segmentation
            progress_callback: Fonction de callback pour suivre la progression
        """
        self.start_url = start_url
        self.max_pages = max_pages
        self.respect_robots = respect_robots
        self.crawl_delay = crawl_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.progress_callback = progress_callback
        
        # Extraire le domaine de l'URL de départ
        parsed_url = urlparse(start_url)
        self.domain = parsed_url.netloc
        self.scheme = parsed_url.scheme
        self.base_url = f"{self.scheme}://{self.domain}"
        
        # Initialiser le gestionnaire de règles de segmentation
        self.segment_rules_file = segment_rules_file or "app/data/segment_rules.json"
        self.segment_manager = SegmentRuleManager(self.segment_rules_file)
        
        # Initialiser les ensembles pour le suivi des URLs
        self.urls_to_crawl = {self.start_url}  # URLs à crawler
        self.crawled_urls = set()  # URLs déjà crawlées
        self.failed_urls = set()  # URLs qui ont échoué
        
        # Liste des patterns d'URLs à exclure du crawl
        self.exclude_patterns = exclude_patterns or []
        
        # Liste des patterns d'URLs à exclure du plan de maillage (mais pas du crawl)
        self.exclude_from_linking_patterns = exclude_from_linking_patterns or []
        
        # Données extraites
        self.pages_data: List[Dict[str, Any]] = []
        self.links_data: List[Dict[str, str]] = []
        
        # Statistiques
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_pages": 0,
            "successful_pages": 0,
            "failed_pages": 0,
            "total_links": 0,
            "internal_links": 0,
            "external_links": 0
        }
        
        logger.info(f"Crawler initialisé pour le domaine: {self.domain}")
        logger.info(f"URL de départ: {self.start_url}")
        logger.info(f"Nombre maximum de pages: {self.max_pages}")
    
    async def crawl(self) -> Tuple[str, str]:
        """
        Lance le processus de crawling et retourne les chemins des fichiers générés.
        
        Returns:
            Tuple contenant les chemins des fichiers de contenu et de liens générés
        """
        self.stats["start_time"] = datetime.now()
        logger.info(f"Début du crawl à {self.stats['start_time']}")
        
        if self.progress_callback:
            await self.progress_callback("Initialisation du crawl...", 0, self.max_pages)
        
        # Vérifier les robots.txt si nécessaire
        if self.respect_robots:
            disallowed_patterns = self._get_robots_txt_rules()
            logger.info(f"Règles robots.txt récupérées: {len(disallowed_patterns)} motifs interdits")
        else:
            disallowed_patterns = []
        
        # Boucle principale de crawling
        pbar = tqdm(total=self.max_pages, desc="Crawling")
        while self.urls_to_crawl and len(self.crawled_urls) < self.max_pages:
            # Prendre la prochaine URL à crawler
            url = self.urls_to_crawl.pop()
            
            # Vérifier si l'URL est autorisée par robots.txt
            if self.respect_robots and self._is_url_disallowed(url, disallowed_patterns):
                logger.info(f"URL ignorée (robots.txt): {url}")
                continue
                
            # Vérifier si l'URL correspond à un pattern d'exclusion
            if self._is_url_excluded(url):
                logger.info(f"URL ignorée (pattern d'exclusion): {url}")
                continue
            
            # Crawler la page
            success, page_data, new_links = await self._crawl_page(url)
            
            if success:
                self.stats["successful_pages"] += 1
                self.pages_data.append(page_data)
                
                # Ajouter les nouvelles URLs à crawler
                for link in new_links:
                    if link not in self.crawled_urls and link not in self.urls_to_crawl:
                        self.urls_to_crawl.add(link)
                
                # Mettre à jour la progression
                pbar.update(1)
                if self.progress_callback:
                    await self.progress_callback(
                        f"Crawling en cours... ({len(self.crawled_urls)}/{self.max_pages})",
                        len(self.crawled_urls),
                        self.max_pages
                    )
            
            # Respecter le délai entre les requêtes
            await asyncio.sleep(self.crawl_delay)
        
        pbar.close()
        self.stats["end_time"] = datetime.now()
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        logger.info(f"Crawl terminé en {duration:.2f} secondes")
        logger.info(f"Pages crawlées: {len(self.crawled_urls)}")
        logger.info(f"Pages réussies: {self.stats['successful_pages']}")
        logger.info(f"Pages échouées: {len(self.failed_urls)}")
        
        # Générer les fichiers de données
        if self.progress_callback:
            await self.progress_callback("Génération des fichiers de données...", 0, 2)
        
        content_file_path = await self._generate_content_file()
        if self.progress_callback:
            await self.progress_callback("Fichier de contenu généré", 1, 2)
        
        links_file_path = await self._generate_links_file()
        if self.progress_callback:
            await self.progress_callback("Fichier de liens généré", 2, 2)
        
        return content_file_path, links_file_path
    
    async def _crawl_page(self, url: str) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Crawle une page et extrait son contenu et ses liens.
        
        Args:
            url: URL de la page à crawler
            
        Returns:
            Tuple contenant:
            - Un booléen indiquant si le crawl a réussi
            - Les données de la page (titre, contenu, etc.)
            - La liste des nouveaux liens internes trouvés
        """
        logger.info(f"Crawling de la page: {url}")
        
        # Marquer l'URL comme crawlée
        self.crawled_urls.add(url)
        self.stats["total_pages"] += 1
        
        # Initialiser les données de la page
        page_data = {
            "url": url,
            "original_url": url,  # Conserver l'URL originale en cas de redirection
            "title": "",
            "h1": "",
            "content": "",
            "segment": "",
            "status_code": 0,
            "crawl_time": datetime.now().isoformat()
        }
        
        new_links = []
        
        # Essayer de récupérer la page avec des tentatives en cas d'échec
        for attempt in range(self.max_retries):
            try:
                # Suivre automatiquement les redirections (allow_redirects=True par défaut)
                response = requests.get(
                    url,
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout,
                    allow_redirects=True
                )
                
                # Si l'URL a été redirigée, enregistrer l'URL finale
                if response.url != url:
                    logger.info(f"Redirection: {url} -> {response.url}")
                    # Mettre à jour l'URL dans les données de la page
                    page_data["original_url"] = url
                    page_data["url"] = response.url
                    # Ajouter l'URL redirigée à l'ensemble des URLs crawlées
                    self.crawled_urls.add(response.url)
                
                page_data["status_code"] = response.status_code
                
                if response.status_code != 200:
                    logger.warning(f"Statut HTTP non-200 pour {url}: {response.status_code}")
                    self.failed_urls.add(url)
                    # Ne pas ajouter les pages avec un code différent de 200 à self.pages_data
                    return False, page_data, []
                
                # Vérifier le type de contenu
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type.lower():
                    logger.warning(f"Type de contenu non HTML pour {url}: {content_type}")
                    self.failed_urls.add(url)
                    return False, page_data, []
                
                # Extraire le contenu et les liens
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Extraire le titre
                title_tag = soup.find("title")
                page_data["title"] = title_tag.get_text().strip() if title_tag else ""
                
                # Extraire le H1
                h1_tag = soup.find("h1")
                page_data["h1"] = h1_tag.get_text().strip() if h1_tag else ""
                
                # Extraire le contenu principal
                content = self._extract_main_content(soup)
                page_data["content"] = content
                
                # Détecter le segment
                page_data["segment"] = self._detect_segment(url, soup)
                
                # Extraire les liens
                links, internal_links = self._extract_links(soup, url)
                self.stats["total_links"] += len(links)
                self.stats["internal_links"] += len(internal_links)
                self.stats["external_links"] += len(links) - len(internal_links)
                
                # Ajouter les liens à la liste des données de liens
                for link in links:
                    self.links_data.append({
                        "Source": url,
                        "Destination": link["url"],
                        "Texte": link["text"],
                        "Interne": link["is_internal"]
                    })
                
                # Retourner uniquement les URLs internes pour continuer le crawl
                new_links = [link["url"] for link in links if link["is_internal"]]
                
                return True, page_data, new_links
                
            except requests.Timeout:
                logger.warning(f"Timeout pour {url} (tentative {attempt+1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    self.failed_urls.add(url)
            except requests.RequestException as e:
                logger.warning(f"Erreur lors du crawl de {url}: {str(e)} (tentative {attempt+1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    self.failed_urls.add(url)
            
            # Attendre avant de réessayer
            await asyncio.sleep(self.crawl_delay * (attempt + 1))
        
        return False, page_data, []
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extrait le contenu principal d'une page en utilisant des heuristiques.
        
        Args:
            soup: Objet BeautifulSoup de la page
            
        Returns:
            Contenu principal de la page
        """
        # Liste des sélecteurs potentiels pour le contenu principal (du plus au moins spécifique)
        main_content_selectors = [
            "main", 
            "article", 
            "#content", 
            ".content", 
            "#main-content", 
            ".main-content",
            ".entry-content",
            "#post-content",
            ".post-content"
        ]
        
        # Essayer chaque sélecteur
        for selector in main_content_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                # Supprimer les éléments non pertinents
                for element in main_element.select("nav, aside, footer, .sidebar, .widget, script, style, .comments"):
                    element.decompose()
                
                return main_element.get_text(separator=" ", strip=True)
        
        # Si aucun sélecteur ne fonctionne, utiliser le body entier mais nettoyer
        body = soup.find("body")
        if not body:
            return ""
        
        # Supprimer les éléments non pertinents
        for element in body.select("header, nav, aside, footer, .sidebar, .widget, script, style, .comments"):
            element.decompose()
        
        return body.get_text(separator=" ", strip=True)
    
    def _extract_links(self, soup: BeautifulSoup, current_url: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extrait tous les liens d'une page.
        
        Args:
            soup: Objet BeautifulSoup de la page
            current_url: URL de la page en cours
            
        Returns:
            Tuple contenant:
            - Liste de tous les liens (internes et externes)
            - Liste des liens internes uniquement
        """
        links = []
        internal_links = []
        
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            link_text = a_tag.get_text(strip=True)
            
            # Ignorer les liens vides ou les ancres
            if not href or href.startswith("#"):
                continue
            
            # Construire l'URL complète
            absolute_url = urljoin(current_url, href)
            
            # Supprimer les fragments d'URL
            absolute_url = urldefrag(absolute_url)[0]
            
            # Vérifier si c'est un lien interne
            parsed_url = urlparse(absolute_url)
            is_internal = parsed_url.netloc == self.domain
            
            # Ignorer les URLs non HTTP/HTTPS
            if parsed_url.scheme not in ["http", "https"]:
                continue
            
            link_data = {
                "url": absolute_url,
                "text": link_text,
                "is_internal": is_internal
            }
            
            links.append(link_data)
            if is_internal:
                internal_links.append(link_data)
        
        return links, internal_links
    
    def _detect_segment(self, url: str, content: Optional[BeautifulSoup] = None) -> str:
        """
        Détecte le segment (type) de la page en fonction des règles définies par l'utilisateur.
        
        Args:
            url: URL de la page
            content: Contenu HTML parsé (BeautifulSoup) - non utilisé dans cette implémentation
            
        Returns:
            Segment détecté selon les règles définies
        """
        # Utiliser le gestionnaire de règles pour déterminer le segment
        segment = self.segment_manager.get_segment(url)
        logger.debug(f"Segment détecté pour {url}: {segment}")
        return segment
    
    def _get_robots_txt_rules(self) -> List[str]:
        """
        Récupère les règles du fichier robots.txt.
        
        Returns:
            Liste des motifs d'URL interdits
        """
        disallowed_patterns = []
        robots_url = f"{self.base_url}/robots.txt"
        
        try:
            response = requests.get(
                robots_url,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                lines = response.text.split("\n")
                current_agent = None
                
                for line in lines:
                    line = line.strip().lower()
                    
                    if line.startswith("user-agent:"):
                        agent = line.split(":", 1)[1].strip()
                        if agent == "*" or agent in self.user_agent.lower():
                            current_agent = agent
                        else:
                            current_agent = None
                    
                    elif current_agent and line.startswith("disallow:"):
                        pattern = line.split(":", 1)[1].strip()
                        if pattern:
                            disallowed_patterns.append(pattern)
            
            logger.info(f"Règles robots.txt récupérées: {len(disallowed_patterns)} motifs")
        except Exception as e:
            logger.warning(f"Erreur lors de la récupération du robots.txt: {str(e)}")
        
        return disallowed_patterns
    
    def _is_url_disallowed(self, url: str, disallowed_patterns: List[str]) -> bool:
        """
        Vérifie si une URL est interdite par les règles robots.txt.
        
        Args:
            url: URL à vérifier
            disallowed_patterns: Liste des motifs interdits
            
        Returns:
            True si l'URL est interdite, False sinon
        """
        
        for pattern in disallowed_patterns:
            if pattern == "/":  # Tout est interdit
                return True
            elif pattern in url:
                return True
        return False
        
    def _is_url_excluded(self, url: str) -> bool:
        """
        Vérifie si une URL correspond à un pattern d'exclusion configuré.
        
        Args:
            url: URL à vérifier
            
        Returns:
            True si l'URL doit être exclue, False sinon
        """
        for pattern in self.exclude_patterns:
            # Si le pattern commence par regex:, le traiter comme une expression régulière
            if pattern.startswith("regex:"):
                regex_pattern = pattern[6:]
                try:
                    if re.search(regex_pattern, url):
                        return True
                except re.error as e:
                    logger.warning(f"Expression régulière invalide '{regex_pattern}': {str(e)}")
            # Sinon, vérifier si le pattern est contenu dans l'URL
            elif pattern in url:
                return True
        return False
        
    def _is_url_excluded_from_linking(self, url: str) -> bool:
        """
        Vérifie si une URL correspond à un pattern d'exclusion du plan de maillage.
        Ces URLs sont crawlées mais ne sont pas incluses dans le plan de maillage.
        
        Args:
            url: URL à vérifier
            
        Returns:
            True si l'URL doit être exclue du plan de maillage, False sinon
        """
        for pattern in self.exclude_from_linking_patterns:
            # Si le pattern commence par regex:, le traiter comme une expression régulière
            if pattern.startswith("regex:"):
                regex_pattern = pattern[6:]
                try:
                    if re.search(regex_pattern, url):
                        return True
                except re.error as e:
                    logger.warning(f"Expression régulière invalide '{regex_pattern}': {str(e)}")
            # Sinon, vérifier si le pattern est contenu dans l'URL
            elif pattern in url:
                return True
        return False
    
    async def _generate_content_file(self) -> str:
        """
        Génère un fichier Excel avec le contenu des pages crawlées.
        Ne conserve que les pages avec un code HTTP 200.
        
        Returns:
            Chemin du fichier généré
        """
        # Créer un DataFrame à partir des données des pages
        # Toutes les pages dans self.pages_data devraient déjà avoir un code 200
        # mais on vérifie quand même pour plus de sécurité
        data = []
        pages_with_200 = [page for page in self.pages_data if page["status_code"] == 200]
        
        for page in pages_with_200:
            data.append({
                "Adresse": page["url"],
                "Segments": page["segment"],
                "Extracteur 1 1": page["content"],
                "Titre": page["title"],
                "H1": page["h1"],
                "Status": page["status_code"]
            })
            
        # Journaliser le nombre de pages conservées
        logger.info(f"Pages conservées pour le maillage (code 200): {len(data)} sur {len(self.pages_data)} pages crawlées")
        
        # Vérifier s'il y a des pages avec un code différent de 200 dans self.pages_data
        non_200_pages = [page for page in self.pages_data if page["status_code"] != 200]
        if non_200_pages:
            logger.warning(f"Attention: {len(non_200_pages)} pages avec un code différent de 200 ont été trouvées dans self.pages_data et ont été filtrées.")
            # Afficher les détails des pages non-200 pour le débogage
            for page in non_200_pages:
                logger.warning(f"  - URL: {page['url']}, Code: {page['status_code']}")
        
        df = pd.DataFrame(data)
        
        # Créer le dossier de sortie si nécessaire
        os.makedirs("app/uploads/content", exist_ok=True)
        
        # Générer un nom de fichier unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crawled_content_{self.domain}_{timestamp}.xlsx"
        filepath = os.path.join("app/uploads/content", filename)
        
        # Sauvegarder le DataFrame dans un fichier Excel
        df.to_excel(filepath, index=False)
        logger.info(f"Fichier de contenu généré: {filepath}")
        
        return filepath
    
    async def _generate_links_file(self) -> str:
        """
        Génère un fichier Excel avec les liens entre les pages crawlées.
        Ne conserve que les liens entre pages avec un code HTTP 200.
        
        Returns:
            Chemin du fichier généré
        """
        # Créer un ensemble des URLs avec un code 200
        # Toutes les pages dans self.pages_data devraient déjà avoir un code 200
        # mais on vérifie quand même pour plus de sécurité
        valid_urls = {page["url"] for page in self.pages_data if page["status_code"] == 200}
        
        # Afficher un log détaillé des URLs valides
        logger.info(f"Nombre d'URLs valides (code 200): {len(valid_urls)}")
        
        # Vérifier si des URLs dans les liens ne sont pas dans valid_urls
        all_link_urls = set()
        for link in self.links_data:
            all_link_urls.add(link["Source"])
            all_link_urls.add(link["Destination"])
        
        invalid_urls = all_link_urls - valid_urls
        if invalid_urls:
            logger.warning(f"Attention: {len(invalid_urls)} URLs dans les liens ne sont pas des pages valides (code 200).")
            # Limiter l'affichage à 10 URLs maximum pour éviter de surcharger les logs
            for url in list(invalid_urls)[:10]:
                logger.warning(f"  - URL invalide dans les liens: {url}")
            if len(invalid_urls) > 10:
                logger.warning(f"  - ... et {len(invalid_urls) - 10} autres URLs invalides")
        
        # Filtrer les liens pour ne garder que les liens internes entre pages valides (code 200)
        filtered_links = []
        
        for link in self.links_data:
            source_url = link["Source"]
            target_url = link["Destination"]
            
            # Ne garder que les liens entre pages valides (code 200) et non exclues du plan de maillage
            if (link["Interne"] and source_url in valid_urls and target_url in valid_urls
                and not self._is_url_excluded_from_linking(source_url) 
                and not self._is_url_excluded_from_linking(target_url)):
                filtered_links.append({
                    "Source": source_url,
                    "Destination": target_url,
                    "Texte": link["Texte"]
                })
                
        # Compter le nombre d'URLs exclues du plan de maillage
        excluded_from_linking_urls = {url for url in valid_urls if self._is_url_excluded_from_linking(url)}
        
        # Journaliser le nombre de liens conservés et d'URLs exclues du plan de maillage
        logger.info(f"Liens conservés pour le maillage (entre pages avec code 200): {len(filtered_links)} sur {len(self.links_data)} liens détectés")
        if excluded_from_linking_urls:
            logger.info(f"URLs exclues du plan de maillage: {len(excluded_from_linking_urls)} URLs")
            # Afficher les 10 premières URLs exclues pour information
            for url in list(excluded_from_linking_urls)[:10]:
                logger.info(f"  - URL exclue du plan de maillage: {url}")
            if len(excluded_from_linking_urls) > 10:
                logger.info(f"  - ... et {len(excluded_from_linking_urls) - 10} autres URLs exclues")
        
        df = pd.DataFrame(filtered_links)
        
        # Créer le dossier de sortie si nécessaire
        os.makedirs("app/uploads/links", exist_ok=True)
        
        # Générer un nom de fichier unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crawled_links_{self.domain}_{timestamp}.xlsx"
        filepath = os.path.join("app/uploads/links", filename)
        
        # Sauvegarder le DataFrame dans un fichier Excel
        df.to_excel(filepath, index=False)
        logger.info(f"Fichier de liens généré: {filepath}")
        
        return filepath
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du crawl.
        
        Returns:
            Dictionnaire des statistiques
        """
        if self.stats["start_time"] and self.stats["end_time"]:
            duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
            self.stats["duration_seconds"] = duration
            self.stats["pages_per_second"] = self.stats["total_pages"] / duration if duration > 0 else 0
        
        return self.stats
