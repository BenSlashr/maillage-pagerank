"""
Routes API pour l'intégration avec Google Search Console.
"""
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
import os
from datetime import datetime, timedelta

from app.services.google_auth import (
    get_auth_url, exchange_code, is_authenticated,
    get_site_list, get_search_analytics, format_search_analytics_to_excel
)

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


@router.get("/auth", response_model=GoogleAuthResponse)
async def google_auth():
    """
    Génère une URL d'authentification pour Google OAuth.
    """
    auth_url, state = get_auth_url()
    return {"auth_url": auth_url, "state": state}


@router.get("/callback")
async def google_callback(code: str = Query(...), state: str = Query(...)):
    """
    Callback pour l'authentification Google OAuth.
    """
    try:
        exchange_code(code, state)
        return RedirectResponse(url="/gsc")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Erreur lors de l'authentification: {str(e)}"}
        )


@router.get("/status")
async def google_status():
    """
    Vérifie le statut d'authentification Google.
    """
    return {"authenticated": is_authenticated()}


@router.get("/sites", response_model=GoogleSiteList)
async def google_sites():
    """
    Récupère la liste des propriétés Search Console.
    """
    if not is_authenticated():
        return {"sites": [], "authenticated": False}
    
    try:
        sites = get_site_list()
        return {"sites": sites, "authenticated": True}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des sites: {str(e)}")


@router.post("/analytics")
async def google_analytics(request: GoogleAnalyticsRequest):
    """
    Récupère les données d'analyse de recherche pour une propriété.
    """
    if not is_authenticated():
        raise HTTPException(status_code=401, detail="Non authentifié. Veuillez vous connecter à Google.")
    
    try:
        analytics_data = get_search_analytics(
            request.site_url,
            request.start_date,
            request.end_date,
            request.dimensions
        )
        
        # Formater les données pour l'export
        formatted_data = format_search_analytics_to_excel(analytics_data, request.dimensions)
        
        return {"data": formatted_data, "total": len(formatted_data)}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des données: {str(e)}")


@router.post("/export-to-file")
async def export_analytics_to_file(request: GoogleAnalyticsRequest):
    """
    Exporte les données d'analyse de recherche vers un fichier CSV.
    """
    if not is_authenticated():
        raise HTTPException(status_code=401, detail="Non authentifié. Veuillez vous connecter à Google.")
    
    try:
        # Récupérer les données d'analyse
        analytics_data = get_search_analytics(
            request.site_url,
            request.start_date,
            request.end_date,
            request.dimensions
        )
        
        # Formater les données
        formatted_data = format_search_analytics_to_excel(analytics_data, request.dimensions)
        
        if not formatted_data:
            raise HTTPException(status_code=404, detail="Aucune donnée trouvée pour cette période")
        
        # Créer un DataFrame pandas
        df = pd.DataFrame(formatted_data)
        
        # Créer le répertoire de sortie si nécessaire
        os.makedirs("app/data/gsc", exist_ok=True)
        
        # Générer un nom de fichier basé sur la date
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        site_name = request.site_url.replace('sc-domain:', '').replace('http://', '').replace('https://', '').replace('/', '_')
        filename = f"gsc_data_{site_name}_{timestamp}.csv"
        filepath = f"app/data/gsc/{filename}"
        
        # Exporter vers CSV
        df.to_csv(filepath, index=False)
        
        return {"filename": filename, "path": filepath, "row_count": len(formatted_data)}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'export des données: {str(e)}")
