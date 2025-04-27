"""
Routes API pour l'intégration avec Google Search Console.
Version personnalisée qui évite les problèmes de sérialisation/désérialisation des dates.
"""
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
import os
from datetime import datetime, timedelta

from app.services.custom_google_auth import CustomGoogleAuth

router = APIRouter(prefix="/api/google", tags=["google"])


class GoogleAuthResponse(BaseModel):
    """Modèle de réponse pour l'authentification Google."""
    auth_url: str
    state: str


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
async def google_auth(request: Request):
    """
    Génère une URL d'authentification pour Google OAuth.
    """
    try:
        print("Début de la génération de l'URL d'authentification...")
        auth_url, state = CustomGoogleAuth.get_auth_url(request)
        print(f"URL d'authentification générée avec succès: {auth_url[:30]}...")
        return {"auth_url": auth_url, "state": state}
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
async def google_callback(request: Request, code: str = Query(...), state: str = Query(...)):
    """
    Callback pour l'authentification Google OAuth.
    """
    try:
        # Échanger le code contre un token
        token_data = CustomGoogleAuth.exchange_code(request, code, state)
        
        # Rediriger vers la page GSC avec un paramètre pour forcer le rafraîchissement
        return RedirectResponse(url="/gsc?auth=success")
    except Exception as e:
        print(f"Erreur dans le callback OAuth: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Erreur lors de l'authentification: {str(e)}"}
        )


@router.get("/logout")
async def logout(request: Request):
    """
    Déconnecte l'utilisateur de Google.
    """
    CustomGoogleAuth.logout(request)
    return RedirectResponse(url="/gsc")


@router.get("/status")
async def google_status(request: Request):
    """
    Vérifie le statut d'authentification Google.
    """
    return {"authenticated": CustomGoogleAuth.is_authenticated(request)}


@router.get("/sites", response_model=GoogleSiteList)
async def google_sites(request: Request):
    """
    Récupère la liste des propriétés Search Console.
    """
    if not CustomGoogleAuth.is_authenticated(request):
        return {"sites": [], "authenticated": False}
    
    try:
        print("Début de la récupération des sites...")
        sites = CustomGoogleAuth.get_site_list(request)
        print(f"Sites récupérés avec succès: {len(sites)} sites")
        return {"sites": sites, "authenticated": True}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des sites: {str(e)}")


@router.post("/analytics")
async def google_analytics(request: Request, analytics_request: GoogleAnalyticsRequest):
    """
    Récupère les données d'analyse de recherche pour une propriété.
    """
    if not CustomGoogleAuth.is_authenticated(request):
        raise HTTPException(status_code=401, detail="Non authentifié. Veuillez vous connecter à Google.")
    
    try:
        analytics_data = CustomGoogleAuth.get_search_analytics(
            request,
            analytics_request.site_url,
            analytics_request.start_date,
            analytics_request.end_date,
            analytics_request.dimensions
        )
        
        # Formater les données pour l'export
        formatted_data = CustomGoogleAuth.format_search_analytics_to_excel(analytics_data, analytics_request.dimensions)
        
        return {"data": formatted_data, "total": len(formatted_data)}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des données: {str(e)}")


@router.post("/export-to-file")
async def export_analytics_to_file(request: Request, analytics_request: GoogleAnalyticsRequest):
    """
    Exporte les données d'analyse de recherche vers un fichier CSV.
    """
    if not CustomGoogleAuth.is_authenticated(request):
        raise HTTPException(status_code=401, detail="Non authentifié. Veuillez vous connecter à Google.")
    
    try:
        # Récupérer les données d'analyse
        analytics_data = CustomGoogleAuth.get_search_analytics(
            request,
            analytics_request.site_url,
            analytics_request.start_date,
            analytics_request.end_date,
            analytics_request.dimensions
        )
        
        # Formater les données
        formatted_data = CustomGoogleAuth.format_search_analytics_to_excel(analytics_data, analytics_request.dimensions)
        
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
