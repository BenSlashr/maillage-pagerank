"""
Routes API pour la configuration des identifiants Google.
"""
from typing import Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Form, UploadFile, File
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import os

from app.services.google_credentials_manager import (
    save_credentials_from_file, save_credentials_manual,
    get_credentials_info, delete_credentials, has_credentials,
    mask_client_id
)

router = APIRouter(prefix="/api/google", tags=["google-config"])


class CredentialsResponse(BaseModel):
    """Modèle de réponse pour les informations d'identifiants."""
    client_id: str
    project_id: str
    success: bool
    message: str


@router.post("/upload-credentials")
async def upload_credentials(request: Request, credentials_file: UploadFile = File(...)):
    """
    Télécharge et sauvegarde les identifiants Google à partir d'un fichier.
    """
    if not credentials_file.filename.endswith('.json'):
        raise HTTPException(
            status_code=400,
            detail="Le fichier doit être au format JSON."
        )
    
    try:
        credentials_info = save_credentials_from_file(credentials_file)
        
        # Stocker le message de succès dans la session
        request.session["credentials_message"] = "Identifiants Google sauvegardés avec succès."
        request.session["credentials_success"] = True
        
        # Rediriger vers la page de configuration
        return RedirectResponse(url="/google-config", status_code=303)
    except HTTPException as e:
        # Stocker le message d'erreur dans la session
        request.session["credentials_message"] = str(e.detail)
        request.session["credentials_success"] = False
        
        # Rediriger vers la page de configuration
        return RedirectResponse(url="/google-config", status_code=303)
    except Exception as e:
        # Stocker le message d'erreur dans la session
        request.session["credentials_message"] = f"Erreur lors de la sauvegarde des identifiants: {str(e)}"
        request.session["credentials_success"] = False
        
        # Rediriger vers la page de configuration
        return RedirectResponse(url="/google-config", status_code=303)


@router.post("/save-credentials")
async def save_credentials(
    request: Request,
    client_id: str = Form(...),
    client_secret: str = Form(...),
    project_id: Optional[str] = Form(None)
):
    """
    Sauvegarde les identifiants Google saisis manuellement.
    """
    try:
        credentials_info = save_credentials_manual(client_id, client_secret, project_id)
        
        # Stocker le message de succès dans la session
        request.session["credentials_message"] = "Identifiants Google sauvegardés avec succès."
        request.session["credentials_success"] = True
        
        # Rediriger vers la page de configuration
        return RedirectResponse(url="/google-config", status_code=303)
    except HTTPException as e:
        # Stocker le message d'erreur dans la session
        request.session["credentials_message"] = str(e.detail)
        request.session["credentials_success"] = False
        
        # Rediriger vers la page de configuration
        return RedirectResponse(url="/google-config", status_code=303)
    except Exception as e:
        # Stocker le message d'erreur dans la session
        request.session["credentials_message"] = f"Erreur lors de la sauvegarde des identifiants: {str(e)}"
        request.session["credentials_success"] = False
        
        # Rediriger vers la page de configuration
        return RedirectResponse(url="/google-config", status_code=303)


@router.post("/delete-credentials")
async def remove_credentials(request: Request):
    """
    Supprime les identifiants Google configurés.
    """
    success = delete_credentials()
    
    if success:
        # Stocker le message de succès dans la session
        request.session["credentials_message"] = "Identifiants Google supprimés avec succès."
        request.session["credentials_success"] = True
    else:
        # Stocker le message d'information dans la session
        request.session["credentials_message"] = "Aucun identifiant Google à supprimer."
        request.session["credentials_success"] = False
    
    # Rediriger vers la page de configuration
    return RedirectResponse(url="/google-config", status_code=303)


@router.get("/status-credentials")
async def get_credentials_status():
    """
    Vérifie si les identifiants Google sont configurés.
    """
    credentials_configured = has_credentials()
    
    return {"has_credentials": credentials_configured}
