"""
Module de gestion simplifiée de l'authentification Google OAuth 2.0.
Utilise l'approche classique avec stockage de token dans un fichier.
"""
import os
import json
from typing import Dict, Optional, Tuple, Any
from datetime import datetime

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from fastapi import HTTPException, Request

# Définition des scopes d'accès requis pour l'API Search Console
SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']

# Chemins des fichiers
APP_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
CONFIG_DIR = os.path.join(APP_ROOT, 'config')
TOKEN_DIR = os.path.join(APP_ROOT, 'tokens')
CLIENT_SECRETS_FILE = os.path.join(CONFIG_DIR, 'client_secret.json')
TOKEN_FILE = os.path.join(TOKEN_DIR, 'token.json')

# URL de base pour les redirections
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

# Créer les répertoires s'ils n'existent pas
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(TOKEN_DIR, exist_ok=True)

def get_auth_url() -> str:
    """
    Génère l'URL d'authentification OAuth 2.0 pour Google.
    
    Returns:
        str: L'URL d'authentification
    """
    try:
        # Vérifier si le fichier de configuration client existe
        if not os.path.exists(CLIENT_SECRETS_FILE):
            raise HTTPException(
                status_code=400, 
                detail="Configuration Google non trouvée. Veuillez configurer vos identifiants Google."
            )
        
        # Créer le flow OAuth
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            redirect_uri=f"{BASE_URL}/api/google/callback"
        )
        
        # Générer l'URL d'authentification
        auth_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        
        return auth_url
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de la génération de l'URL d'authentification: {str(e)}"
        )

def exchange_code(code: str) -> Dict[str, Any]:
    """
    Échange le code d'autorisation contre un token d'accès.
    
    Args:
        code: Le code d'autorisation reçu de Google
        
    Returns:
        Dict[str, Any]: Les informations d'authentification, y compris le token d'accès
    """
    try:
        # Créer le flow OAuth
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            redirect_uri=f"{BASE_URL}/api/google/callback"
        )
        
        # Échanger le code contre un token
        flow.fetch_token(code=code)
        
        # Extraire les informations du token
        credentials = flow.credentials
        token_data = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }
        
        # Sauvegarder le token dans un fichier
        with open(TOKEN_FILE, 'w') as f:
            json.dump(token_data, f)
        
        return token_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de l'échange du code d'autorisation: {str(e)}"
        )

def get_credentials() -> Optional[Credentials]:
    """
    Récupère les credentials d'authentification Google depuis le fichier de token.
    
    Returns:
        Optional[Credentials]: Les credentials d'authentification ou None si non disponibles
    """
    if not os.path.exists(TOKEN_FILE):
        return None
    
    try:
        with open(TOKEN_FILE, 'r') as f:
            token_data = json.load(f)
        
        # Créer les credentials à partir des données du token
        credentials = Credentials(
            token=token_data.get('token'),
            refresh_token=token_data.get('refresh_token'),
            token_uri=token_data.get('token_uri'),
            client_id=token_data.get('client_id'),
            client_secret=token_data.get('client_secret'),
            scopes=token_data.get('scopes')
        )
        
        return credentials
        
    except Exception as e:
        print(f"Erreur lors de la récupération des credentials: {str(e)}")
        # En cas d'erreur, supprimer le fichier de token
        if os.path.exists(TOKEN_FILE):
            os.remove(TOKEN_FILE)
        return None

def is_authenticated() -> bool:
    """
    Vérifie si l'utilisateur est authentifié auprès de Google.
    
    Returns:
        bool: True si l'utilisateur est authentifié, False sinon
    """
    return get_credentials() is not None

def get_search_console_service():
    """
    Crée et retourne un service pour l'API Search Console.
    
    Returns:
        Resource: Le service Search Console
    """
    credentials = get_credentials()
    
    if not credentials:
        raise HTTPException(
            status_code=401, 
            detail="Non authentifié. Veuillez vous connecter à Google."
        )
    
    try:
        return build('searchconsole', 'v1', credentials=credentials)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de la création du service Search Console: {str(e)}"
        )

def logout() -> bool:
    """
    Déconnecte l'utilisateur de Google en supprimant le fichier de token.
    
    Returns:
        bool: True si la déconnexion a réussi, False sinon
    """
    if os.path.exists(TOKEN_FILE):
        try:
            os.remove(TOKEN_FILE)
            return True
        except Exception:
            return False
    return True
