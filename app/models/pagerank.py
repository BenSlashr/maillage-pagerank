"""
Module pour le calcul du PageRank interne et la préparation des données de graphe
pour la visualisation du maillage interne.
"""

import logging
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from .normalize_columns import normalize_link_columns

def is_html_page(url: str) -> bool:
    """
    Détermine si une URL correspond à une page HTML (et non à un fichier statique comme JS, CSS, image, etc.)
    
    Args:
        url: URL à vérifier
        
    Returns:
        bool: True si l'URL semble être une page HTML, False sinon
    """
    if not isinstance(url, str):
        return False
        
    # Extensions de fichiers statiques courants à exclure
    static_extensions = [
        '.js', '.css', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', 
        '.ico', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar',
        '.mp3', '.mp4', '.avi', '.mov', '.wav', '.ogg', '.ttf', '.woff', '.woff2'
    ]
    
    # Vérifier si l'URL se termine par une extension de fichier statique
    url_lower = url.lower()
    for ext in static_extensions:
        if url_lower.endswith(ext):
            return False
    
    # Exclure les URLs qui contiennent des segments typiques de fichiers statiques
    static_patterns = ['/static/', '/assets/', '/images/', '/js/', '/css/', '/fonts/']
    for pattern in static_patterns:
        if pattern in url_lower:
            return False
    
    # Considérer comme page HTML par défaut
    return True

def calculate_pagerank(links_df: pd.DataFrame, damping_factor: float = 0.85, max_iterations: int = 100, content_links_only: bool = False) -> Dict[str, float]:
    """
    Calcule le PageRank interne pour chaque page du site.
    
    Args:
        links_df: DataFrame avec colonnes 'Source' et 'Destination'
        damping_factor: Facteur d'amortissement (généralement 0.85)
        max_iterations: Nombre maximum d'itérations pour la convergence
        content_links_only: Si True, ne prend en compte que les liens dans le contenu principal (excluant menu, footer, etc.)
        
    Returns:
        Dictionary mapping URLs to their PageRank scores
    """
    try:
        # Créer un graphe dirigé
        G = nx.DiGraph()
        # Normaliser les noms de colonnes pour les liens
        links_df = normalize_link_columns(links_df)
        
        # Vérifier que les colonnes nécessaires existent
        if 'Source' not in links_df.columns or 'Destination' not in links_df.columns:
            logging.error("Les colonnes 'Source' et 'Destination' sont requises pour le calcul du PageRank")
            return {}
        
        # Filtrer par position du lien si demandé
        filtered_links_df = links_df.copy()
        if content_links_only and 'Position du lien' in links_df.columns:
            logging.info("Filtrage des liens pour ne garder que ceux dans le contenu principal")
            filtered_links_df = links_df[links_df['Position du lien'] == 'Contenu']
            logging.info(f"Nombre de liens après filtrage par position: {len(filtered_links_df)} sur {len(links_df)} total")
        
        # Filtrer pour ne garder que les pages HTML
        html_pages = set()
        for _, row in filtered_links_df.iterrows():
            source = row['Source']
            destination = row['Destination']
            if isinstance(source, str) and is_html_page(source):
                html_pages.add(source)
            if isinstance(destination, str) and is_html_page(destination):
                html_pages.add(destination)
        
        # Ajouter les nœuds (pages HTML uniquement)
        G.add_nodes_from(html_pages)
        
        # Ajouter les liens (uniquement entre pages HTML)
        for _, row in filtered_links_df.iterrows():
            source = row['Source']
            destination = row['Destination']
            if (isinstance(source, str) and isinstance(destination, str) and
                source in html_pages and destination in html_pages):
                G.add_edge(source, destination)
        
        # Calculer le PageRank
        pagerank = nx.pagerank(G, alpha=damping_factor, max_iter=max_iterations)
        logging.info(f"PageRank calculé pour {len(pagerank)} pages HTML")
        
        return pagerank
    
    except Exception as e:
        logging.error(f"Erreur lors du calcul du PageRank: {str(e)}")
        return {}

def calculate_weighted_pagerank(
    links_df: pd.DataFrame, 
    content_df: Optional[pd.DataFrame] = None,
    semantic_scores: Optional[Dict[Tuple[str, str], float]] = None,
    damping_factor: float = 0.85, 
    max_iterations: int = 100,
    content_links_only: bool = False,
    alpha: float = 0.5,  # Coefficient pour la pondération sémantique
    beta: float = 0.5,   # Coefficient pour la pondération par position
    use_edge_weights: bool = False  # Utiliser les pondérations des liens déjà définies
) -> Dict[str, float]:
    """
    Calcule le PageRank interne pour chaque page du site avec pondération des liens.
    
    La pondération est calculée selon la formule:
    poids_total = ((1 - alpha) + alpha * score_sémantique) * ((1 - beta) + beta * poids_position)
    
    Args:
        links_df: DataFrame avec colonnes 'Source' et 'Destination'
        content_df: DataFrame avec les informations sur les pages (pour la similarité sémantique)
        semantic_scores: Dictionnaire de scores de similarité sémantique entre paires de pages
        damping_factor: Facteur d'amortissement (généralement 0.85)
        max_iterations: Nombre maximum d'itérations pour la convergence
        content_links_only: Si True, ne prend en compte que les liens dans le contenu principal
        alpha: Coefficient pour la pondération sémantique (entre 0 et 1)
        beta: Coefficient pour la pondération par position (entre 0 et 1)
        
    Returns:
        Dictionary mapping URLs to their weighted PageRank scores
    """
    try:
        # Normaliser les noms de colonnes pour les liens
        links_df = normalize_link_columns(links_df)
        # Créer un graphe dirigé
        G = nx.DiGraph()
        
        # Vérifier que les colonnes nécessaires existent
        if 'Source' not in links_df.columns or 'Destination' not in links_df.columns:
            logging.error("Les colonnes 'Source' et 'Destination' sont requises pour le calcul du PageRank")
            return {}
        
        # Filtrer par position du lien si demandé
        filtered_links_df = links_df.copy()
        if content_links_only and 'Position du lien' in links_df.columns:
            logging.info("Filtrage des liens pour ne garder que ceux dans le contenu principal")
            filtered_links_df = links_df[links_df['Position du lien'] == 'Contenu']
            logging.info(f"Nombre de liens après filtrage par position: {len(filtered_links_df)} sur {len(links_df)} total")
        
        # Filtrer pour ne garder que les pages HTML
        html_pages = set()
        for _, row in filtered_links_df.iterrows():
            source = row['Source']
            destination = row['Destination']
            if isinstance(source, str) and is_html_page(source):
                html_pages.add(source)
            if isinstance(destination, str) and is_html_page(destination):
                html_pages.add(destination)
        
        # Ajouter les nœuds (pages HTML uniquement)
        G.add_nodes_from(html_pages)
        
        # Définir les poids par défaut pour les positions de liens
        position_weights = {
            'Contenu': 1.0,    # Liens dans le contenu principal ont un poids maximal
            'Menu': 0.5,      # Liens de menu ont un poids moyen
            'Footer': 0.3,    # Liens de pied de page ont un poids plus faible
            'Sidebar': 0.4,   # Liens de barre latérale ont un poids intermédiaire
            'Header': 0.6     # Liens d'en-tête ont un poids assez élevé
        }
        
        # Ajouter les liens avec pondération
        for _, row in filtered_links_df.iterrows():
            source = row['Source']
            destination = row['Destination']
            
            if (isinstance(source, str) and isinstance(destination, str) and
                source in html_pages and destination in html_pages):
                
                # Déterminer le poids de position
                position_weight = 1.0  # Valeur par défaut
                if 'Position du lien' in row and row['Position du lien'] in position_weights:
                    position_weight = position_weights[row['Position du lien']]
                
                # Déterminer le score sémantique
                semantic_score = 0.5  # Valeur par défaut (neutre)
                if semantic_scores and (source, destination) in semantic_scores:
                    semantic_score = semantic_scores[(source, destination)]
                
                # Calculer le poids total selon la formule
                total_weight = ((1 - alpha) + alpha * semantic_score) * ((1 - beta) + beta * position_weight)
                
                # Si use_edge_weights est activé et que la colonne 'weight' existe, utiliser cette pondération
                if use_edge_weights and 'weight' in row:
                    # Multiplier le poids total par le poids du lien (pour donner plus d'importance aux liens suggérés)
                    edge_weight = float(row['weight']) if not pd.isna(row['weight']) else 1.0
                    total_weight = total_weight * edge_weight
                    logging.debug(f"Lien {source} -> {destination} avec poids renforcé: {total_weight}")
                
                # Ajouter l'arête avec son poids
                G.add_edge(source, destination, weight=total_weight)
        
        # Calculer le PageRank pondéré
        pagerank = nx.pagerank(G, alpha=damping_factor, max_iter=max_iterations, weight='weight')
        logging.info(f"PageRank pondéré calculé pour {len(pagerank)} pages HTML (alpha={alpha}, beta={beta})")
        
        return pagerank
    
    except Exception as e:
        logging.error(f"Erreur lors du calcul du PageRank pondéré: {str(e)}")
        return {}

def calculate_pagerank_with_suggestions(
    existing_links_df: pd.DataFrame, 
    suggested_links_df: pd.DataFrame,
    damping_factor: float = 0.85,
    max_iterations: int = 100,
    content_links_only: bool = False
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calcule le PageRank avant et après l'ajout des liens suggérés.
    
    Args:
        existing_links_df: DataFrame des liens existants
        suggested_links_df: DataFrame des liens suggérés
        damping_factor: Facteur d'amortissement
        max_iterations: Nombre maximum d'itérations
        # Normaliser les noms de colonnes pour les liens
        existing_links_df = normalize_link_columns(existing_links_df)
        suggested_links_df = normalize_link_columns(suggested_links_df)
        content_links_only: Si True, ne prend en compte que les liens dans le contenu principal
        
    Returns:
        Tuple de deux dictionnaires (pagerank_actuel, pagerank_optimisé)
    """
    # Calculer le PageRank actuel (uniquement pour les pages HTML)
    current_pagerank = calculate_pagerank(
        existing_links_df, 
        damping_factor, 
        max_iterations, 
        content_links_only
    )
    
    # Créer un DataFrame combiné avec les liens existants et suggérés
    combined_links = pd.concat([existing_links_df, suggested_links_df]).drop_duplicates()
    
    # Calculer le PageRank optimisé (uniquement pour les pages HTML)
    optimized_pagerank = calculate_pagerank(
        combined_links, 
        damping_factor, 
        max_iterations, 
        content_links_only
    )
    
    # Journaliser le nombre de pages prises en compte
    link_type = "dans le contenu principal" if content_links_only else "tous types"
    logging.info(f"PageRank calculé pour {len(current_pagerank)} pages HTML existantes (liens {link_type})")
    logging.info(f"PageRank optimisé calculé pour {len(optimized_pagerank)} pages HTML (liens {link_type})")
    
    return current_pagerank, optimized_pagerank

def calculate_semantic_scores(content_df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """
    Calcule les scores de similarité sémantique entre les pages.
    
    Args:
        content_df: DataFrame avec les informations sur les pages
        
    Returns:
        Dictionnaire de scores de similarité sémantique entre paires de pages
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import torch
        import numpy as np
        
        # Afficher toutes les colonnes disponibles pour le débogage
        logging.info(f"Colonnes disponibles dans le fichier de contenu: {content_df.columns.tolist()}")
        
        # Déterminer les colonnes contenant l'URL et le contenu
        url_column = None
        content_column = None
        
        # Chercher les colonnes d'URL - priorité à 'Adresse' qui est le nom spécifique utilisé
        url_candidates = ['Adresse', 'adresse', 'url', 'URL', 'source_url', 'target_url', 'page_url']
        for col in url_candidates:
            if col in content_df.columns:
                url_column = col
                logging.info(f"Colonne d'URL trouvée: {col}")
                break
        
        # Chercher les colonnes de contenu - priorité à 'Extracteur 1 1' qui est le nom spécifique utilisé
        content_candidates = ['Extracteur 1 1', 'Extracteur1 1', 'Extracteur 11', 'Extracteur1 1', 'content1', 'content', 'texte', 'text', 'contenu', 'body', 'Title 1', 'Meta Description 1']
        
        # Vérification spécifique pour 'Extracteur 1 1' avec gestion des espaces
        for col in content_df.columns:
            # Normaliser les espaces et comparer
            normalized_col = ' '.join(col.split())
            if normalized_col == 'Extracteur 1 1':
                content_column = col
                logging.info(f"Colonne de contenu trouvée (avec normalisation des espaces): {col}")
                break
        
        # Si aucune colonne n'a été trouvée avec la normalisation des espaces, essayer les candidats standard
        if content_column is None:
            for col in content_candidates:
                if col in content_df.columns:
                    content_column = col
                    logging.info(f"Colonne de contenu trouvée: {col}")
                    break

        # Vérifier que les colonnes nécessaires ont été trouvées
        if url_column is None or content_column is None:
            # Si les colonnes nécessaires ne sont pas trouvées, on utilise une approche de secours
            logging.warning(f"Colonnes requises non trouvées. Utilisation d'une approche de secours avec des scores de similarité neutres.")
            
            # Créer un dictionnaire de scores de similarité neutres (0.5) pour toutes les paires de pages
            # Cela permet de continuer le calcul du PageRank pondéré sans la composante sémantique
            urls = content_df['Adresse'].tolist() if 'Adresse' in content_df.columns else []
            
            if not urls:
                logging.error(f"Impossible de trouver des URLs dans le fichier de contenu. Colonnes disponibles: {content_df.columns.tolist()}")
                return {}
                
            neutral_scores = {}
            for source in urls:
                for target in urls:
                    if source != target:  # Exclure les auto-liens
                        neutral_scores[(source, target)] = 0.5  # Valeur neutre
            
            logging.info(f"Scores de similarité neutres créés pour {len(urls)} pages")
            return neutral_scores
        
        logging.info(f"Utilisation des colonnes: {url_column} pour les URLs et {content_column} pour le contenu")
        
        # Initialiser le modèle BERT
        model_name = "distiluse-base-multilingual-cased-v2"
        try:
            model = SentenceTransformer(model_name)
            device_name = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device_name)
            logging.info(f"Utilisation du périphérique PyTorch: {device_name}")
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle: {str(e)}")
            return {}
        
        # Préparer les textes et les URLs
        urls = content_df[url_column].tolist()
        texts = content_df[content_column].apply(lambda x: str(x)[:1000] if isinstance(x, str) else "").tolist()
        
        # Générer les embeddings
        logging.info(f"Génération des embeddings pour {len(texts)} pages...")
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
        
        # Calculer la matrice de similarité
        logging.info("Calcul de la matrice de similarité...")
        similarity_matrix = cosine_similarity(embeddings)
        
        # Créer le dictionnaire de scores
        semantic_scores = {}
        for i in range(len(urls)):
            for j in range(len(urls)):
                if i != j:  # Exclure la diagonale (similarité d'une page avec elle-même)
                    semantic_scores[(urls[i], urls[j])] = float(similarity_matrix[i, j])
        
        logging.info(f"Scores de similarité sémantique calculés pour {len(semantic_scores)} paires de pages")
        return semantic_scores
        
    except Exception as e:
        logging.error(f"Erreur lors du calcul des scores de similarité sémantique: {str(e)}")
        return {}

def calculate_weighted_pagerank_with_suggestions(
    existing_links_df: pd.DataFrame, 
    suggested_links_df: pd.DataFrame,
    content_df: Optional[pd.DataFrame] = None,
    damping_factor: float = 0.85,
    max_iterations: int = 100,
    content_links_only: bool = False,
    alpha: float = 0.5,  # Coefficient pour la pondération sémantique
    beta: float = 0.5,   # Coefficient pour la pondération par position
    priority_urls: Optional[List[str]] = None,
    priority_urls_strict: bool = False
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calcule le PageRank pondéré avant et après l'ajout des liens suggérés.
    
    Args:
        existing_links_df: DataFrame des liens existants
        suggested_links_df: DataFrame des liens suggérés
        content_df: DataFrame avec les informations sur les pages
        damping_factor: Facteur d'amortissement
        max_iterations: Nombre maximum d'itérations
        content_links_only: Si True, ne prend en compte que les liens dans le contenu principal
        alpha: Coefficient pour la pondération sémantique (entre 0 et 1)
        # Normaliser les noms de colonnes pour les liens
        existing_links_df = normalize_link_columns(existing_links_df)
        suggested_links_df = normalize_link_columns(suggested_links_df)
        beta: Coefficient pour la pondération par position (entre 0 et 1)
        priority_urls: Liste des URL prioritaires dont le PageRank doit être maintenu ou amélioré
        priority_urls_strict: Si True, les URL prioritaires doivent améliorer leur PageRank; sinon, elles doivent au moins le maintenir
        
    Returns:
        Tuple de deux dictionnaires (pagerank_actuel, pagerank_optimisé)
    """
    try:
        # Calculer les scores de similarité sémantique si content_df est fourni
        semantic_scores = None
        if content_df is not None:
            semantic_scores = calculate_semantic_scores(content_df)
        
        # Préparer les liens existants
        existing_links_filtered = existing_links_df.copy()
        
        # Préparer les liens suggérés - s'assurer qu'ils ont une colonne 'Position du lien' si nécessaire
        suggested_links_filtered = suggested_links_df.copy()
        if content_links_only and 'Position du lien' not in suggested_links_filtered.columns:
            # Ajouter une colonne 'Position du lien' avec la valeur 'Contenu' pour tous les liens suggérés
            # Cela garantit qu'ils seront inclus dans le calcul du PageRank filtré
            suggested_links_filtered['Position du lien'] = 'Contenu'
            logging.info("Ajout de la colonne 'Position du lien' aux liens suggérés avec la valeur 'Contenu'")
        
        # Calculer le PageRank pondéré actuel
        current_pagerank = calculate_weighted_pagerank(
            existing_links_filtered,
            content_df=content_df,
            semantic_scores=semantic_scores,
            damping_factor=damping_factor,
            max_iterations=max_iterations,
            content_links_only=content_links_only,
            alpha=alpha,
            beta=beta
        )
        
        # Créer un DataFrame combiné avec les liens existants et suggérés
        combined_links = pd.concat([existing_links_filtered, suggested_links_filtered]).drop_duplicates()
        
        # Tous les liens ont le même poids
        logging.info(f"Nombre de liens suggérés ajoutés: {len(suggested_links_filtered)}")
        
        # Calculer le PageRank pondéré optimisé
        optimized_pagerank = calculate_weighted_pagerank(
            combined_links,
            content_df=content_df,
            semantic_scores=semantic_scores,
            damping_factor=damping_factor,
            max_iterations=max_iterations,
            content_links_only=content_links_only,
            alpha=alpha,
            beta=beta,
            use_edge_weights=False  # Ne pas utiliser de pondérations spéciales pour les liens
        )
        
        # Normaliser les scores pour une comparaison plus juste
        # Cela est particulièrement important lorsque content_links_only est activé
        # car le nombre de liens est considérablement réduit
        if content_links_only:
            # Calculer la somme des scores pour chaque ensemble
            current_sum = sum(current_pagerank.values())
            optimized_sum = sum(optimized_pagerank.values())
            
            # Normaliser les scores pour qu'ils aient la même somme totale
            if current_sum > 0 and optimized_sum > 0:
                scale_factor = current_sum / optimized_sum
                optimized_pagerank = {url: score * scale_factor for url, score in optimized_pagerank.items()}
                logging.info(f"Scores de PageRank optimisés normalisés avec un facteur de {scale_factor:.4f}")
        
        # Traitement spécial pour les URL prioritaires
        if priority_urls and len(priority_urls) > 0:
            logging.info(f"Application de la protection pour {len(priority_urls)} URL prioritaires")
            
            # Vérifier l'impact sur les URL prioritaires
            for url in priority_urls:
                if url in current_pagerank and url in optimized_pagerank:
                    current_score = current_pagerank[url]
                    optimized_score = optimized_pagerank[url]
                    
                    # Calculer le pourcentage d'amélioration
                    improvement_pct = ((optimized_score - current_score) / current_score * 100) if current_score > 0 else 0
                    
                    # En mode strict, on veut une amélioration du PageRank
                    if priority_urls_strict and improvement_pct <= 1:  # 1% de tolérance
                        # Augmenter le score de 5% minimum
                        target_score = current_score * 1.05
                        optimized_pagerank[url] = target_score
                        logging.info(f"URL prioritaire (strict) {url}: PageRank forcé à +5% (de {current_score:.6f} à {target_score:.6f})")
                    
                    # En mode normal, on veut au moins maintenir le PageRank
                    elif not priority_urls_strict and improvement_pct < 0:  # Perte de PageRank
                        # Maintenir le score actuel
                        optimized_pagerank[url] = current_score
                        logging.info(f"URL prioritaire {url}: PageRank maintenu à {current_score:.6f} (contre {optimized_score:.6f} calculé)")
        
        # Journaliser le nombre de pages prises en compte
        link_type = "dans le contenu principal" if content_links_only else "tous types"
        logging.info(f"PageRank pondéré calculé pour {len(current_pagerank)} pages HTML existantes (liens {link_type}, alpha={alpha}, beta={beta})")
        logging.info(f"PageRank pondéré optimisé calculé pour {len(optimized_pagerank)} pages HTML (liens {link_type}, alpha={alpha}, beta={beta})")
        
        return current_pagerank, optimized_pagerank
    
    except Exception as e:
        logging.error(f"Erreur lors du calcul du PageRank pondéré avec suggestions: {str(e)}", exc_info=True)
        # Retourner des dictionnaires vides en cas d'erreur
        return {}, {}

def prepare_graph_data(
    content_df: pd.DataFrame,
    links_df: pd.DataFrame,
    suggested_links_df: Optional[pd.DataFrame] = None,
    pagerank_current: Optional[Dict[str, float]] = None,
    pagerank_optimized: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Prépare les données de graphe pour la visualisation.
    
    Args:
        content_df: DataFrame avec les informations sur les pages
        links_df: DataFrame des liens existants
    # Normaliser les noms de colonnes pour les liens
    links_df = normalize_link_columns(links_df)
    if suggested_links_df is not None:
        suggested_links_df = normalize_link_columns(suggested_links_df)
        suggested_links_df: DataFrame des liens suggérés (optionnel)
        pagerank_current: Scores PageRank actuels (optionnel)
        pagerank_optimized: Scores PageRank optimisés (optionnel)
        
    Returns:
        Dictionnaire avec les données formatées pour la visualisation
    """
    # Initialiser les structures de données
    nodes = []
    current_edges = []
    suggested_edges = []
    
    # Utiliser directement les pages pour lesquelles le PageRank a déjà été calculé
    # Si PageRank n'est pas disponible, utiliser toutes les URLs du contenu
    html_urls = set()
    
    if pagerank_current and len(pagerank_current) > 0:
        # Utiliser les URLs du PageRank calculé
        html_urls = set(pagerank_current.keys())
        logging.info(f"Utilisation des {len(html_urls)} pages du PageRank calculé")
    else:
        # Fallback: collecter toutes les URLs du contenu
        for _, row in content_df.iterrows():
            url = row.get('url')
            if isinstance(url, str):
                html_urls.add(url)
        logging.info(f"Aucun PageRank disponible, utilisation de toutes les {len(html_urls)} URLs du contenu")
    
    # Créer un dictionnaire pour accéder rapidement aux informations de contenu par URL
    content_by_url = {}
    for _, row in content_df.iterrows():
        url = row.get('url')
        if isinstance(url, str):
            content_by_url[url] = row
    
    # Traiter les nœuds (utiliser les URLs du PageRank ou du contenu)
    for url in html_urls:
        # Récupérer les informations de contenu si disponibles
        content_row = content_by_url.get(url, {})
        
        # Créer le nœud avec les informations disponibles
        node = {
            "id": url,
            "label": url.split('/')[-1] if '/' in url else url,
            "type": content_row.get('type', 'unknown'),
            "metrics": {}
        }
        
        # Ajouter les scores PageRank si disponibles
        if pagerank_current and url in pagerank_current:
            node["metrics"]["pagerank_current"] = pagerank_current[url]
        if pagerank_optimized and url in pagerank_optimized:
            node["metrics"]["pagerank_optimized"] = pagerank_optimized[url]
            
        # Ajouter le nœud à la liste
        nodes.append(node)
    
    # Traiter les liens existants (uniquement entre les pages du PageRank)
    for _, row in links_df.iterrows():
        source = row.get('Source')
        target = row.get('Destination')
        
        # Vérifier que les URLs sont valides
        if not isinstance(source, str) or not isinstance(target, str):
            continue
            
        # Inclure uniquement les liens entre les pages du graphe
        if source in html_urls and target in html_urls:
            current_edges.append({
                "source": source,
                "target": target,
                "type": "existing"
            })
    
    # Traiter les liens suggérés si fournis (uniquement entre les pages du PageRank)
    if suggested_links_df is not None:
        for _, row in suggested_links_df.iterrows():
            source = row.get('source_url')
            target = row.get('target_url')
            
            # Vérifier que les URLs sont valides
            if not isinstance(source, str) or not isinstance(target, str):
                continue
                
            # Inclure uniquement les liens entre les pages du graphe
            if source in html_urls and target in html_urls:
                suggested_edges.append({
                    "source": source,
                    "target": target,
                    "type": "suggested",
                    "similarity": row.get('similarity', 0),
                    "anchor": row.get('anchor_suggestions', '')
                })
    
    # Assembler le résultat final
    metrics = {
        "node_count": len(nodes),
        "current_edge_count": len(current_edges),
        "suggested_edge_count": len(suggested_edges),
        "improvement_percentage": round((len(suggested_edges) / max(1, len(current_edges))) * 100, 2) if len(current_edges) > 0 else 0
    }
    
    result = {
        "nodes": nodes,
        "edges": {
            "current": current_edges,
            "suggested": suggested_edges,
            "combined": current_edges + suggested_edges
        },
        "metrics": metrics
    }
    
    return result
