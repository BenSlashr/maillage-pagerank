"""
Module de gestion de l'authentification Google OAuth 2.0 et de l'accès à l'API Search Console.
Version avec gestion des sessions pour environnement multi-utilisateurs.
"""
import os
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import secrets

import httplib2
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from fastapi import HTTPException, Request, Response, Depends
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

# Définition des scopes d'accès requis pour l'API Search Console
SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']

# Chemin du fichier de configuration client OAuth
# Utiliser un chemin absolu plus explicite
APP_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
CONFIG_DIR = os.path.join(APP_ROOT, 'config')
CLIENT_SECRETS_FILE = os.path.join(CONFIG_DIR, 'client_secret.json')

print(f"Chemin absolu du fichier de configuration: {CLIENT_SECRETS_FILE}")
print(f"Le fichier existe: {os.path.exists(CLIENT_SECRETS_FILE)}")

# Clé secrète pour les sessions - à remplacer par une variable d'environnement en production
SECRET_KEY = secrets.token_hex(32)

# URL de base pour les redirections - à configurer selon l'environnement
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")


def get_auth_url(request: Request) -> Tuple[str, str]:
    """
    Génère l'URL d'authentification OAuth 2.0 pour Google.
    
    Args:
        request: La requête HTTP pour accéder à la session
        
    Returns:
        Tuple[str, str]: Un tuple contenant (auth_url, state) où auth_url est l'URL 
                        d'authentification et state est un identifiant unique pour la session.
    """
    try:
        # Vérifier si le fichier de configuration client existe
        print(f"Vérification du fichier de configuration: {CLIENT_SECRETS_FILE}")
        if not os.path.exists(CLIENT_SECRETS_FILE):
            print(f"Fichier de configuration introuvable: {CLIENT_SECRETS_FILE}")
            raise HTTPException(
                status_code=500, 
                detail="Fichier de configuration client OAuth manquant. Veuillez configurer l'API Google Cloud."
            )
        
        print(f"Fichier de configuration trouvé: {CLIENT_SECRETS_FILE}")
        
        try:
            # Créer le flow OAuth
            print("Création du flow OAuth...")
            flow = Flow.from_client_secrets_file(
                CLIENT_SECRETS_FILE,
                scopes=SCOPES,
                redirect_uri=f"{BASE_URL}/api/google/callback"
            )
            print(f"Flow OAuth créé avec succès, redirect_uri: {BASE_URL}/api/google/callback")
            
            # Générer l'URL d'authentification
            print("Génération de l'URL d'authentification...")
            auth_url, state = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent'
            )
            print(f"URL d'authentification générée: {auth_url[:50]}...")
            
            # Sauvegarder l'état dans la session
            request.session["oauth_state"] = state
            print(f"État OAuth sauvegardé dans la session: {state[:10]}...")
            
            return auth_url, state
        except Exception as inner_e:
            print(f"Erreur lors de la création du flow OAuth: {str(inner_e)}")
            raise
    
    except Exception as e:
        print(f"Erreur générale lors de la génération de l'URL d'authentification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de l'URL d'authentification: {str(e)}")


def exchange_code(request: Request, code: str, state: str) -> Dict[str, Any]:
    """
    Échange le code d'autorisation contre un token d'accès.
    
    Args:
        request: La requête HTTP pour accéder à la session
        code: Le code d'autorisation reçu de Google
        state: L'identifiant d'état pour vérifier la session
        
    Returns:
        Dict[str, Any]: Les informations d'authentification, y compris le token d'accès
    """
    try:
        # Vérifier si l'état correspond à celui stocké dans la session
        session_state = request.session.get("oauth_state")
        if not session_state or session_state != state:
            raise HTTPException(status_code=400, detail="État de session invalide")
        
        # Créer le flow OAuth
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            redirect_uri=f"{BASE_URL}/api/google/callback",
            state=state
        )
        
        # Échanger le code contre un token
        flow.fetch_token(code=code)
        credentials = flow.credentials
        
        # Sauvegarder les credentials dans la session
        # Stocker uniquement les informations nécessaires sans l'expiry pour éviter les problèmes
        # L'expiry sera recalculée automatiquement lors de la récupération
        token_data = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
            # Ne pas stocker l'expiry du tout
        }
        
        # Stocker dans la session
        request.session["google_token"] = token_data
        
        # Supprimer l'état de la session
        if "oauth_state" in request.session:
            del request.session["oauth_state"]
        
        return token_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'échange du code d'autorisation: {str(e)}")


def get_credentials(request: Request) -> Optional[Credentials]:
    """
    Récupère les credentials d'authentification Google depuis la session.
    
    Args:
        request: La requête HTTP pour accéder à la session
        
    Returns:
        Optional[Credentials]: Les credentials d'authentification ou None si non disponibles
    """
    token_data = request.session.get("google_token")
    
    if not token_data:
        return None
    
    try:
        # Ne pas essayer de convertir l'expiry, nous ne la stockons plus
        # Nous laissons la bibliothèque Google OAuth2 gérer l'expiration automatiquement
        
        credentials = Credentials.from_authorized_user_info(token_data, SCOPES)
        
        # Vérifier si les credentials sont expirés et les rafraîchir si nécessaire
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(httplib2.Http())
            
            # Mettre à jour la session sans stocker l'expiry
            token_data = {
                'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes
                # Ne pas stocker l'expiry du tout
            }
            
            request.session["google_token"] = token_data
        
        return credentials
    
    except Exception as e:
        print(f"Erreur lors de la récupération des credentials: {str(e)}")
        # En cas d'erreur, supprimer le token de la session
        if "google_token" in request.session:
            del request.session["google_token"]
        return None


def is_authenticated(request: Request) -> bool:
    """
    Vérifie si l'utilisateur est authentifié auprès de Google.
    
    Args:
        request: La requête HTTP pour accéder à la session
        
    Returns:
        bool: True si l'utilisateur est authentifié, False sinon
    """
    # Vérifier si le token est présent dans la session
    if "google_token" not in request.session:
        return False
        
    # Essayer de récupérer les credentials valides
    try:
        credentials = get_credentials(request)
        if not credentials:
            return False
            
        # Vérifier si les credentials sont expirés
        if credentials.expired and not credentials.refresh_token:
            return False
            
        return True
    except Exception as e:
        print(f"Erreur lors de la vérification de l'authentification: {str(e)}")
        return False


def get_search_console_service(request: Request):
    """
    Crée et retourne un service pour l'API Search Console.
    
    Args:
        request: La requête HTTP pour accéder à la session
        
    Returns:
        Resource: Le service Search Console
    """
    credentials = get_credentials(request)
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Non authentifié. Veuillez vous connecter à Google.")
    
    try:
        return build('searchconsole', 'v1', credentials=credentials)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la création du service Search Console: {str(e)}")


def get_site_list(request: Request) -> List[Dict[str, str]]:
    """
    Récupère la liste des propriétés Search Console de l'utilisateur.
    
    Args:
        request: La requête HTTP pour accéder à la session
        
    Returns:
        List[Dict[str, str]]: Liste des propriétés avec leur URL et leur nom
    """
    print("Début de la fonction get_site_list...")
    service = get_search_console_service(request)
    
    try:
        print("Appel à l'API Google pour récupérer les sites...")
        sites = service.sites().list().execute()
        print(f"Réponse de l'API reçue avec {len(sites.get('siteEntry', []))} sites")
        site_list = []
        
        for site_entry in sites.get('siteEntry', []):
            try:
                # Récupérer les données du site avec vérification de type
                site_url = site_entry.get('siteUrl', '')
                permission_level = site_entry.get('permissionLevel', '')
                
                # Vérification du type de site_url
                if not isinstance(site_url, str):
                    print(f"ATTENTION: site_url n'est pas une chaîne mais un {type(site_url)}: {site_url}")
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
                print(f"Erreur lors du traitement d'un site: {str(e)}")
                # Continuer avec le site suivant
                continue
        
        print(f"Liste des sites traitée avec succès: {len(site_list)} sites valides")
        return site_list
    
    except HttpError as e:
        error_content = json.loads(e.content.decode('utf-8'))
        error_message = error_content.get('error', {}).get('message', str(e))
        raise HTTPException(status_code=e.resp.status, detail=f"Erreur API Google: {error_message}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des sites: {str(e)}")


def get_search_analytics(request: Request, site_url: str, start_date: str, end_date: str, dimensions: List[str]) -> Dict[str, Any]:
    """
    Récupère les données d'analyse de recherche pour une propriété spécifique.
    
    Args:
        request: La requête HTTP pour accéder à la session
        site_url: L'URL de la propriété Search Console
        start_date: Date de début au format YYYY-MM-DD
        end_date: Date de fin au format YYYY-MM-DD
        dimensions: Les dimensions à inclure (page, query, device, country, etc.)
        
    Returns:
        Dict[str, Any]: Les données d'analyse de recherche
    """
    service = get_search_console_service(request)
    
    try:
        # Définir les paramètres de la requête
        request_body = {
            'startDate': start_date,
            'endDate': end_date,
            'dimensions': dimensions,
            'rowLimit': 5000,  # Limite maximale de l'API
            'startRow': 0,
            'searchType': 'web'
        }
        
        # Exécuter la requête
        response = service.searchanalytics().query(siteUrl=site_url, body=request_body).execute()
        
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
        analytics_data: Les données d'analyse de recherche
        dimensions: Les dimensions incluses dans les données
        
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


def logout(request: Request):
    """
    Déconnecte l'utilisateur de Google en supprimant les tokens de la session.
    
    Args:
        request: La requête HTTP pour accéder à la session
    """
    if "google_token" in request.session:
        del request.session["google_token"]
    
    if "oauth_state" in request.session:
        del request.session["oauth_state"]
