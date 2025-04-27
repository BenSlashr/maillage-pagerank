"""
Module de gestion des identifiants Google OAuth 2.0.
"""
import os
import json
import secrets
from typing import Dict, Optional, Tuple, Any
from fastapi import HTTPException, UploadFile

# Chemin du répertoire pour stocker les identifiants
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')

# Chemin du fichier des identifiants
CREDENTIALS_FILE = os.path.join(CONFIG_DIR, 'client_secret.json')

# S'assurer que le répertoire de configuration existe
os.makedirs(CONFIG_DIR, exist_ok=True)


def save_credentials_from_file(file: UploadFile) -> Dict[str, Any]:
    """
    Sauvegarde les identifiants Google à partir d'un fichier téléchargé.
    
    Args:
        file: Le fichier d'identifiants téléchargé
        
    Returns:
        Dict[str, Any]: Les informations d'identifiants sauvegardées
    """
    try:
        # Lire le contenu du fichier
        content = file.file.read()
        credentials_data = json.loads(content.decode('utf-8'))
        
        # Vérifier que le fichier contient les informations nécessaires
        if 'web' not in credentials_data:
            raise HTTPException(
                status_code=400,
                detail="Format de fichier d'identifiants invalide. Assurez-vous d'avoir téléchargé le bon fichier JSON depuis Google Cloud Console."
            )
        
        # Vérifier les champs requis
        web_config = credentials_data['web']
        required_fields = ['client_id', 'client_secret', 'auth_uri', 'token_uri']
        
        for field in required_fields:
            if field not in web_config:
                raise HTTPException(
                    status_code=400,
                    detail=f"Le fichier d'identifiants ne contient pas le champ requis '{field}'."
                )
        
        # Ajouter l'URI de redirection si nécessaire
        base_url = os.environ.get("BASE_URL", "http://localhost:8000")
        redirect_uri = f"{base_url}/api/google/callback"
        
        if 'redirect_uris' not in web_config:
            web_config['redirect_uris'] = [redirect_uri]
        elif redirect_uri not in web_config['redirect_uris']:
            web_config['redirect_uris'].append(redirect_uri)
        
        # Sauvegarder les identifiants
        with open(CREDENTIALS_FILE, 'w') as f:
            json.dump(credentials_data, f, indent=2)
        
        return {
            'client_id': web_config['client_id'],
            'project_id': web_config.get('project_id', 'Non spécifié'),
            'redirect_uris': web_config['redirect_uris']
        }
    
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Le fichier n'est pas un JSON valide."
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la sauvegarde des identifiants: {str(e)}"
        )
    
    finally:
        # Fermer le fichier
        file.file.close()


def save_credentials_manual(client_id: str, client_secret: str, project_id: str = None) -> Dict[str, Any]:
    """
    Sauvegarde les identifiants Google à partir des informations saisies manuellement.
    
    Args:
        client_id: L'identifiant client OAuth 2.0
        client_secret: Le secret client OAuth 2.0
        project_id: L'identifiant du projet Google Cloud (optionnel)
        
    Returns:
        Dict[str, Any]: Les informations d'identifiants sauvegardées
    """
    try:
        # Construire les données d'identifiants
        base_url = os.environ.get("BASE_URL", "http://localhost:8000")
        redirect_uri = f"{base_url}/api/google/callback"
        
        credentials_data = {
            'web': {
                'client_id': client_id,
                'client_secret': client_secret,
                'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
                'token_uri': 'https://oauth2.googleapis.com/token',
                'auth_provider_x509_cert_url': 'https://www.googleapis.com/oauth2/v1/certs',
                'redirect_uris': [redirect_uri]
            }
        }
        
        if project_id:
            credentials_data['web']['project_id'] = project_id
        
        # Sauvegarder les identifiants
        with open(CREDENTIALS_FILE, 'w') as f:
            json.dump(credentials_data, f, indent=2)
        
        return {
            'client_id': client_id,
            'project_id': project_id or 'Non spécifié',
            'redirect_uris': [redirect_uri]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la sauvegarde des identifiants: {str(e)}"
        )


def get_credentials_info() -> Optional[Dict[str, Any]]:
    """
    Récupère les informations sur les identifiants Google configurés.
    
    Returns:
        Optional[Dict[str, Any]]: Les informations d'identifiants ou None si non configurés
    """
    if not os.path.exists(CREDENTIALS_FILE):
        return None
    
    try:
        with open(CREDENTIALS_FILE, 'r') as f:
            credentials_data = json.load(f)
        
        web_config = credentials_data.get('web', {})
        
        return {
            'client_id': web_config.get('client_id'),
            'project_id': web_config.get('project_id', 'Non spécifié'),
            'redirect_uris': web_config.get('redirect_uris', [])
        }
    
    except Exception:
        return None


def delete_credentials() -> bool:
    """
    Supprime les identifiants Google configurés.
    
    Returns:
        bool: True si les identifiants ont été supprimés, False sinon
    """
    if not os.path.exists(CREDENTIALS_FILE):
        return False
    
    try:
        os.remove(CREDENTIALS_FILE)
        return True
    except Exception:
        return False


def has_credentials() -> bool:
    """
    Vérifie si les identifiants Google sont configurés.
    
    Returns:
        bool: True si les identifiants sont configurés, False sinon
    """
    return os.path.exists(CREDENTIALS_FILE)


def mask_client_id(client_id: str) -> str:
    """
    Masque une partie de l'identifiant client pour l'affichage.
    
    Args:
        client_id: L'identifiant client complet
        
    Returns:
        str: L'identifiant client partiellement masqué
    """
    if not client_id:
        return ""
    
    parts = client_id.split('.')
    
    if len(parts) >= 2:
        # Masquer la première partie (avant le premier point)
        prefix = parts[0]
        if len(prefix) > 8:
            masked_prefix = prefix[:4] + '*' * (len(prefix) - 8) + prefix[-4:]
        else:
            masked_prefix = prefix[:2] + '*' * (len(prefix) - 4) + prefix[-2:]
        
        # Reconstruire l'identifiant
        return masked_prefix + '.' + '.'.join(parts[1:])
    
    # Si le format est différent, masquer simplement le milieu
    if len(client_id) > 8:
        return client_id[:4] + '*' * (len(client_id) - 8) + client_id[-4:]
    else:
        return client_id[:2] + '*' * (len(client_id) - 4) + client_id[-2:]
