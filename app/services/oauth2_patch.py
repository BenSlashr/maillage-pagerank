"""
Module pour corriger les problèmes avec la bibliothèque Google OAuth2.
Ce module applique un monkey patch pour corriger l'erreur 'datetime.datetime' object has no attribute 'rstrip'.
"""
import logging
from datetime import datetime
from google.oauth2 import credentials

# Sauvegarde de la fonction originale
original_from_authorized_user_info = credentials.Credentials.from_authorized_user_info

def patched_from_authorized_user_info(cls, info, scopes=None):
    """
    Version patchée de la méthode from_authorized_user_info qui corrige le problème
    avec les objets datetime et la méthode rstrip.
    """
    try:
        # Vérifier si expiry est un objet datetime et le convertir en string si nécessaire
        if "expiry" in info and isinstance(info["expiry"], datetime):
            # Convertir en string au format ISO8601 avec Z
            info = info.copy()  # Créer une copie pour ne pas modifier l'original
            info["expiry"] = info["expiry"].strftime("%Y-%m-%dT%H:%M:%SZ")
            
        # Appeler la fonction originale
        return original_from_authorized_user_info(cls, info, scopes)
    except Exception as e:
        logging.error(f"Erreur dans patched_from_authorized_user_info: {str(e)}")
        # Si expiry cause des problèmes, essayons de le supprimer
        if "expiry" in info:
            info_copy = info.copy()
            info_copy.pop("expiry", None)
            try:
                return original_from_authorized_user_info(cls, info_copy, scopes)
            except Exception as e2:
                logging.error(f"Deuxième erreur dans patched_from_authorized_user_info: {str(e2)}")
                raise e2
        raise e

def apply_patches():
    """
    Applique les monkey patches aux fonctions problématiques.
    """
    # Remplacer la méthode from_authorized_user_info
    credentials.Credentials.from_authorized_user_info = classmethod(patched_from_authorized_user_info)
    logging.info("Monkey patch appliqué à google.oauth2.credentials.Credentials.from_authorized_user_info")
