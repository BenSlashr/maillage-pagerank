def normalize_url(url: str) -> str:
    """Normalise une URL pour les comparaisons"""
    # Supprimer les paramètres d'URL
    url = url.split('?')[0].split('#')[0]
    
    # Supprimer le slash final s'il existe
    if url.endswith('/'):
        url = url[:-1]
        
    # Convertir en minuscules
    url = url.lower()
    
    return url
