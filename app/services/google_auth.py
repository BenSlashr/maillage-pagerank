"""
Module de gestion de l'authentification Google OAuth 2.0 et de l'accès à l'API Search Console.
"""
import os
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import httplib2
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from fastapi import HTTPException

# Définition des scopes d'accès requis pour l'API Search Console
SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']

# Chemin du fichier de configuration client OAuth
CLIENT_SECRETS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'client_secret.json')

# Chemin du répertoire pour stocker les tokens d'authentification
TOKEN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tokens')

# S'assurer que le répertoire des tokens existe
os.makedirs(TOKEN_DIR, exist_ok=True)


def get_auth_url() -> Tuple[str, str]:
    """
    Génère l'URL d'authentification OAuth 2.0 pour Google.
    
    Returns:
        Tuple[str, str]: Un tuple contenant (auth_url, state) où auth_url est l'URL 
                        d'authentification et state est un identifiant unique pour la session.
    """
    try:
        # Vérifier si le fichier de configuration client existe
        if not os.path.exists(CLIENT_SECRETS_FILE):
            raise HTTPException(
                status_code=500, 
                detail="Fichier de configuration client OAuth manquant. Veuillez configurer l'API Google Cloud."
            )
        
        # Créer le flow OAuth
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            redirect_uri="http://localhost:8000/api/google/callback"
        )
        
        # Générer l'URL d'authentification
        auth_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        
        # Sauvegarder l'état pour la vérification ultérieure
        with open(os.path.join(TOKEN_DIR, f"{state}.json"), 'w') as f:
            json.dump({'state': state}, f)
        
        return auth_url, state
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de l'URL d'authentification: {str(e)}")


def exchange_code(code: str, state: str) -> Dict[str, Any]:
    """
    Échange le code d'autorisation contre un token d'accès.
    
    Args:
        code (str): Le code d'autorisation reçu de Google
        state (str): L'identifiant d'état pour vérifier la session
        
    Returns:
        Dict[str, Any]: Les informations d'authentification, y compris le token d'accès
    """
    try:
        # Vérifier si l'état correspond à une session valide
        state_file = os.path.join(TOKEN_DIR, f"{state}.json")
        if not os.path.exists(state_file):
            raise HTTPException(status_code=400, detail="État de session invalide")
        
        # Créer le flow OAuth
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            redirect_uri="http://localhost:8000/api/google/callback",
            state=state
        )
        
        # Échanger le code contre un token
        flow.fetch_token(code=code)
        credentials = flow.credentials
        
        # Sauvegarder les credentials
        token_data = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes,
            'expiry': credentials.expiry.isoformat() if credentials.expiry else None
        }
        
        # Sauvegarder le token dans un fichier
        token_file = os.path.join(TOKEN_DIR, "token.json")
        with open(token_file, 'w') as f:
            json.dump(token_data, f)
        
        # Supprimer le fichier d'état
        os.remove(state_file)
        
        return token_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'échange du code d'autorisation: {str(e)}")


def get_credentials() -> Optional[Credentials]:
    """
    Récupère les credentials d'authentification Google.
    
    Returns:
        Optional[Credentials]: Les credentials d'authentification ou None si non disponibles
    """
    token_file = os.path.join(TOKEN_DIR, "token.json")
    
    if not os.path.exists(token_file):
        return None
    
    try:
        with open(token_file, 'r') as f:
            token_data = json.load(f)
        
        # Ne pas essayer de convertir l'expiry, nous la supprimons pour éviter les problèmes
        if 'expiry' in token_data:
            token_data.pop('expiry', None)
        
        credentials = Credentials.from_authorized_user_info(token_data, SCOPES)
        
        # Vérifier si les credentials sont expirés et les rafraîchir si nécessaire
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(httplib2.Http())
            
            # Mettre à jour le fichier de token sans stocker l'expiry
            token_data = {
                'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes
                # Ne pas stocker l'expiry du tout
            }
            
            with open(token_file, 'w') as f:
                json.dump(token_data, f)
        
        return credentials
    
    except Exception as e:
        print(f"Erreur lors de la récupération des credentials: {str(e)}")
        # En cas d'erreur, supprimer le fichier de token pour forcer une nouvelle authentification
        if os.path.exists(token_file):
            os.remove(token_file)
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
        raise HTTPException(status_code=401, detail="Non authentifié. Veuillez vous connecter à Google.")
    
    try:
        return build('searchconsole', 'v1', credentials=credentials)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la création du service Search Console: {str(e)}")


def get_site_list() -> List[Dict[str, str]]:
    """
    Récupère la liste des propriétés Search Console de l'utilisateur.
    
    Returns:
        List[Dict[str, str]]: Liste des propriétés avec leur URL et leur nom
    """
    service = get_search_console_service()
    
    try:
        sites = service.sites().list().execute()
        site_list = []
        
        for site_entry in sites.get('siteEntry', []):
            try:
                # Récupérer les données du site avec vérification de type
                site_url = site_entry.get('siteUrl', '')
                permission_level = site_entry.get('permissionLevel', '')
                
                # Vérification du type de site_url
                if not isinstance(site_url, str):
                    site_url = str(site_url)
                
                # Ne garder que les propriétés avec un niveau d'accès suffisant
                if permission_level in ['siteOwner', 'siteFullUser']:
                    # Formatage sécurisé du nom du site
                    site_name = site_url
                    if 'sc-domain:' in site_url:
                        site_name = site_name.replace('sc-domain:', '')
                    if 'http://' in site_name:
                        site_name = site_name.replace('http://', '')
                    if 'https://' in site_name:
                        site_name = site_name.replace('https://', '')
                    if site_name.endswith('/'):
                        site_name = site_name[:-1]
                    
                    site_list.append({
                        'url': site_url,
                        'name': site_name
                    })
            except Exception as e:
                # Continuer avec le site suivant
                continue
        
        return site_list
    
    except HttpError as e:
        error_content = json.loads(e.content.decode('utf-8'))
        error_message = error_content.get('error', {}).get('message', str(e))
        raise HTTPException(status_code=e.resp.status, detail=f"Erreur API Google: {error_message}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des sites: {str(e)}")


def get_search_analytics(site_url: str, start_date: str, end_date: str, dimensions: List[str]) -> Dict[str, Any]:
    """
    Récupère les données d'analyse de recherche pour une propriété spécifique.
    
    Args:
        site_url (str): L'URL de la propriété Search Console
        start_date (str): Date de début au format YYYY-MM-DD
        end_date (str): Date de fin au format YYYY-MM-DD
        dimensions (List[str]): Les dimensions à inclure (page, query, device, country, etc.)
        
    Returns:
        Dict[str, Any]: Les données d'analyse de recherche
    """
    service = get_search_console_service()
    
    try:
        # Définir les paramètres de la requête
        request = {
            'startDate': start_date,
            'endDate': end_date,
            'dimensions': dimensions,
            'rowLimit': 5000,  # Limite maximale de l'API
            'startRow': 0,
            'searchType': 'web'
        }
        
        # Exécuter la requête
        response = service.searchanalytics().query(siteUrl=site_url, body=request).execute()
        
        return response
    
    except HttpError as e:
        error_content = json.loads(e.content.decode('utf-8'))
        error_message = error_content.get('error', {}).get('message', str(e))
        raise HTTPException(status_code=e.resp.status, detail=f"Erreur API Google: {error_message}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des données d'analyse: {str(e)}")


def format_search_analytics_to_excel(analytics_data: Dict[str, Any], dimensions: List[str]) -> List[Dict[str, Any]]:
    """
    Formate les données d'analyse de recherche pour l'export Excel.
    
    Args:
        analytics_data (Dict[str, Any]): Les données d'analyse de recherche
        dimensions (List[str]): Les dimensions incluses dans les données
        
    Returns:
        List[Dict[str, Any]]: Liste de dictionnaires formatés pour l'export Excel
    """
    formatted_data = []
    
    for row in analytics_data.get('rows', []):
        item = {}
        
        # Ajouter les dimensions
        for i, dimension in enumerate(dimensions):
            if dimension == 'page':
                item['URL'] = row['keys'][i]
            elif dimension == 'query':
                item['Mot-clé'] = row['keys'][i]
            else:
                item[dimension.capitalize()] = row['keys'][i]
        
        # Ajouter les métriques
        item['Clics'] = row.get('clicks', 0)
        item['Impressions'] = row.get('impressions', 0)
        item['CTR'] = row.get('ctr', 0)
        item['Position'] = row.get('position', 0)
        
        formatted_data.append(item)
    
    return formatted_data
