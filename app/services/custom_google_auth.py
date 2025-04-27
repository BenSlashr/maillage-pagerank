"""
Module de gestion personnalisée de l'authentification Google OAuth 2.0.
Cette implémentation contourne les problèmes liés à la sérialisation/désérialisation des dates.
"""
import os
import json
import secrets
import httplib2
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from fastapi import HTTPException, Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Définition des scopes d'accès requis pour l'API Search Console
SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']

# Chemin du fichier de configuration client OAuth
APP_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
CONFIG_DIR = os.path.join(APP_ROOT, 'config')
CLIENT_SECRETS_FILE = os.path.join(CONFIG_DIR, 'client_secret.json')

# URL de base pour les redirections
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

# Clé secrète pour la session
SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_hex(32))

class CustomGoogleAuth:
    """
    Classe pour gérer l'authentification Google OAuth 2.0 sans utiliser directement les objets Credentials.
    """
    
    @staticmethod
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
            if not os.path.exists(CLIENT_SECRETS_FILE):
                raise HTTPException(
                    status_code=400, 
                    detail="Configuration Google non trouvée. Veuillez configurer vos identifiants Google."
                )
            
            # Créer un identifiant d'état unique pour la session
            state = secrets.token_hex(16)
            
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
                prompt='consent',
                state=state
            )
            
            # Stocker l'état dans la session
            request.session["oauth_state"] = state
            
            return auth_url, state
            
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur lors de la génération de l'URL d'authentification: {str(e)}"
            )
    
    @staticmethod
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
            
            # Extraire les informations essentielles du token
            token_data = {
                'token': flow.credentials.token,
                'refresh_token': flow.credentials.refresh_token,
                'token_uri': flow.credentials.token_uri,
                'client_id': flow.credentials.client_id,
                'client_secret': flow.credentials.client_secret,
                'scopes': flow.credentials.scopes
                # Ne pas stocker l'expiry pour éviter les problèmes
            }
            
            # Stocker dans la session
            request.session["google_token"] = token_data
            
            # Supprimer l'état de la session
            if "oauth_state" in request.session:
                del request.session["oauth_state"]
            
            return token_data
            
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur lors de l'échange du code d'autorisation: {str(e)}"
            )
    
    @staticmethod
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
        
        token_data = request.session.get("google_token")
        
        # Vérifier les champs obligatoires
        required_fields = ['token', 'client_id', 'client_secret']
        for field in required_fields:
            if field not in token_data or not token_data[field]:
                return False
        
        return True
    
    @staticmethod
    def get_search_console_service(request: Request):
        """
        Crée et retourne un service pour l'API Search Console.
        
        Args:
            request: La requête HTTP pour accéder à la session
            
        Returns:
            Resource: Le service Search Console
        """
        if not CustomGoogleAuth.is_authenticated(request):
            raise HTTPException(
                status_code=401, 
                detail="Non authentifié. Veuillez vous connecter à Google."
            )
        
        try:
            # Récupérer les informations du token
            token_data = request.session.get("google_token")
            
            # Créer un objet httplib2.Http pour les requêtes
            http = httplib2.Http()
            
            # Ajouter les en-têtes d'autorisation
            http.add_credentials(
                token_data['client_id'],
                token_data['client_secret']
            )
            
            # Créer le service Search Console
            service = build(
                'searchconsole', 
                'v1', 
                http=http,
                developerKey=token_data['token']
            )
            
            return service
            
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur lors de la création du service Search Console: {str(e)}"
            )
    
    @staticmethod
    def get_site_list(request: Request) -> List[Dict[str, str]]:
        """
        Récupère la liste des propriétés Search Console de l'utilisateur.
        
        Args:
            request: La requête HTTP pour accéder à la session
            
        Returns:
            List[Dict[str, str]]: Liste des propriétés avec leur URL et leur nom
        """
        service = CustomGoogleAuth.get_search_console_service(request)
        
        try:
            sites = service.sites().list().execute()
            site_list = []
            
            if 'siteEntry' in sites:
                for site in sites['siteEntry']:
                    site_url = site.get('siteUrl', '')
                    site_name = site_url
                    
                    # Formater le nom du site pour l'affichage
                    if site_url.startswith('sc-domain:'):
                        site_name = site_url.replace('sc-domain:', '')
                    elif site_url.startswith('https://'):
                        site_name = site_url.replace('https://', '')
                    elif site_url.startswith('http://'):
                        site_name = site_url.replace('http://', '')
                    
                    # Supprimer le slash final si présent
                    if site_name.endswith('/'):
                        site_name = site_name[:-1]
                    
                    site_list.append({
                        'url': site_url,
                        'name': site_name,
                        'permission': site.get('permissionLevel', 'NONE')
                    })
            
            return site_list
            
        except HttpError as e:
            if e.resp.status == 401:
                # Token expiré ou invalide, supprimer de la session
                if "google_token" in request.session:
                    del request.session["google_token"]
                raise HTTPException(
                    status_code=401, 
                    detail="Session expirée. Veuillez vous reconnecter à Google."
                )
            else:
                raise HTTPException(
                    status_code=e.resp.status, 
                    detail=f"Erreur Google API: {str(e)}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur lors de la récupération des sites: {str(e)}"
            )
    
    @staticmethod
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
        service = CustomGoogleAuth.get_search_console_service(request)
        
        try:
            # Préparer la requête
            request_body = {
                'startDate': start_date,
                'endDate': end_date,
                'dimensions': dimensions,
                'rowLimit': 5000,  # Nombre maximum de lignes à récupérer
                'startRow': 0,
                'searchType': 'web'  # Type de recherche (web, image, video, etc.)
            }
            
            # Exécuter la requête
            response = service.searchanalytics().query(
                siteUrl=site_url,
                body=request_body
            ).execute()
            
            return response
            
        except HttpError as e:
            if e.resp.status == 401:
                # Token expiré ou invalide, supprimer de la session
                if "google_token" in request.session:
                    del request.session["google_token"]
                raise HTTPException(
                    status_code=401, 
                    detail="Session expirée. Veuillez vous reconnecter à Google."
                )
            else:
                raise HTTPException(
                    status_code=e.resp.status, 
                    detail=f"Erreur Google API: {str(e)}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur lors de la récupération des données: {str(e)}"
            )
    
    @staticmethod
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
        
        if 'rows' not in analytics_data:
            return formatted_data
        
        for row in analytics_data['rows']:
            item = {}
            
            # Ajouter les dimensions
            for i, dimension in enumerate(dimensions):
                if i < len(row['keys']):
                    item[dimension] = row['keys'][i]
            
            # Ajouter les métriques
            item['clicks'] = row.get('clicks', 0)
            item['impressions'] = row.get('impressions', 0)
            item['ctr'] = row.get('ctr', 0) * 100  # Convertir en pourcentage
            item['position'] = row.get('position', 0)
            
            formatted_data.append(item)
        
        return formatted_data
    
    @staticmethod
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
