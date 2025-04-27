"""
Routes API pour l'intégration avec Google Search Console.
Version simplifiée utilisant un stockage de token dans un fichier.
"""
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
import os
from datetime import datetime, timedelta

from app.services.simple_google_auth import (
    get_auth_url, exchange_code, is_authenticated,
    get_search_console_service, logout
)

router = APIRouter(prefix="/api/google", tags=["google"])


class GoogleSiteList(BaseModel):
    """Modèle de réponse pour la liste des sites GSC."""
    sites: List[Dict[str, str]]
    authenticated: bool


class GoogleAnalyticsRequest(BaseModel):
    """Modèle de requête pour les données d'analyse GSC."""
    site_url: str
    start_date: str
    end_date: str
    dimensions: List[str]


@router.get("/auth")
async def google_auth():
    """
    Génère une URL d'authentification pour Google OAuth.
    """
    try:
        print("Début de la génération de l'URL d'authentification...")
        auth_url = get_auth_url()
        print(f"URL d'authentification générée avec succès: {auth_url[:30]}...")
        return {"auth_url": auth_url}
    except HTTPException as e:
        print(f"HTTPException dans google_auth: {e.detail}")
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail}
        )
    except Exception as e:
        print(f"Exception générale dans google_auth: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Erreur lors de la génération de l'URL d'authentification: {str(e)}"}
        )


@router.get("/callback")
async def google_callback(code: str = Query(...)):
    """
    Callback pour l'authentification Google OAuth.
    """
    try:
        # Échanger le code contre un token
        exchange_code(code)
        
        # Rediriger vers la page GSC avec un paramètre pour forcer le rafraîchissement
        return RedirectResponse(url="/gsc?auth=success")
    except Exception as e:
        print(f"Erreur dans le callback OAuth: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Erreur lors de l'authentification: {str(e)}"}
        )


@router.get("/logout")
async def google_logout():
    """
    Déconnecte l'utilisateur de Google.
    """
    logout()
    return RedirectResponse(url="/gsc")


@router.get("/status")
async def google_status():
    """
    Vérifie le statut d'authentification Google.
    """
    return {"authenticated": is_authenticated()}


@router.get("/status-credentials")
async def get_credentials_status():
    """
    Vérifie si les identifiants Google sont configurés.
    """
    from app.services.google_credentials_manager import has_credentials
    credentials_configured = has_credentials()
    
    return {"has_credentials": credentials_configured}


@router.get("/sites", response_model=GoogleSiteList)
async def google_sites():
    """
    Récupère la liste des propriétés Search Console.
    """
    if not is_authenticated():
        return {"sites": [], "authenticated": False}
    
    try:
        print("Début de la récupération des sites...")
        service = get_search_console_service()
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
        
        print(f"Sites récupérés avec succès: {len(site_list)} sites")
        return {"sites": site_list, "authenticated": True}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des sites: {str(e)}")


@router.post("/analytics")
async def google_analytics(analytics_request: GoogleAnalyticsRequest):
    """
    Récupère les données d'analyse de recherche pour une propriété.
    """
    if not is_authenticated():
        raise HTTPException(status_code=401, detail="Non authentifié. Veuillez vous connecter à Google.")
    
    try:
        service = get_search_console_service()
        
        # Préparer la requête
        request_body = {
            'startDate': analytics_request.start_date,
            'endDate': analytics_request.end_date,
            'dimensions': analytics_request.dimensions,
            'rowLimit': 5000,
            'startRow': 0,
            'searchType': 'web'
        }
        
        # Exécuter la requête
        analytics_data = service.searchanalytics().query(
            siteUrl=analytics_request.site_url,
            body=request_body
        ).execute()
        
        # Formater les données pour l'export
        formatted_data = []
        
        if 'rows' in analytics_data:
            for row in analytics_data['rows']:
                item = {}
                
                # Ajouter les dimensions
                for i, dimension in enumerate(analytics_request.dimensions):
                    if i < len(row['keys']):
                        item[dimension] = row['keys'][i]
                
                # Ajouter les métriques
                item['clicks'] = row.get('clicks', 0)
                item['impressions'] = row.get('impressions', 0)
                item['ctr'] = row.get('ctr', 0) * 100  # Convertir en pourcentage
                item['position'] = row.get('position', 0)
                
                formatted_data.append(item)
        
        return {"data": formatted_data, "total": len(formatted_data)}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des données: {str(e)}")


@router.post("/export-to-file")
async def export_analytics_to_file(analytics_request: GoogleAnalyticsRequest):
    """
    Exporte les données d'analyse de recherche vers un fichier CSV.
    """
    if not is_authenticated():
        raise HTTPException(status_code=401, detail="Non authentifié. Veuillez vous connecter à Google.")
    
    try:
        service = get_search_console_service()
        
        # Préparer la requête
        request_body = {
            'startDate': analytics_request.start_date,
            'endDate': analytics_request.end_date,
            'dimensions': analytics_request.dimensions,
            'rowLimit': 5000,
            'startRow': 0,
            'searchType': 'web'
        }
        
        # Exécuter la requête
        analytics_data = service.searchanalytics().query(
            siteUrl=analytics_request.site_url,
            body=request_body
        ).execute()
        
        # Formater les données
        formatted_data = []
        
        if 'rows' in analytics_data:
            for row in analytics_data['rows']:
                item = {}
                
                # Ajouter les dimensions
                for i, dimension in enumerate(analytics_request.dimensions):
                    if i < len(row['keys']):
                        item[dimension] = row['keys'][i]
                
                # Ajouter les métriques
                item['clicks'] = row.get('clicks', 0)
                item['impressions'] = row.get('impressions', 0)
                item['ctr'] = row.get('ctr', 0) * 100  # Convertir en pourcentage
                item['position'] = row.get('position', 0)
                
                formatted_data.append(item)
        
        if not formatted_data:
            raise HTTPException(status_code=404, detail="Aucune donnée trouvée pour cette période")
        
        # Créer un DataFrame pandas
        df = pd.DataFrame(formatted_data)
        
        # Créer le répertoire de sortie si nécessaire
        os.makedirs("app/data/gsc", exist_ok=True)
        
        # Générer un nom de fichier basé sur la date
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        site_name = analytics_request.site_url.replace('sc-domain:', '').replace('http://', '').replace('https://', '').replace('/', '_')
        filename = f"gsc_data_{site_name}_{timestamp}.csv"
        filepath = f"app/data/gsc/{filename}"
        
        # Exporter vers CSV
        df.to_csv(filepath, index=False)
        
        return {"filename": filename, "path": filepath, "row_count": len(formatted_data)}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'export des données: {str(e)}")
