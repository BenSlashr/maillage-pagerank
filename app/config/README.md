# Configuration Google Search Console

Ce dossier doit contenir votre fichier de configuration OAuth pour Google Search Console.

## Étapes pour configurer l'authentification Google

1. Accédez à la [Console Google Cloud](https://console.cloud.google.com/)
2. Créez un nouveau projet ou sélectionnez un projet existant
3. Activez l'API Google Search Console dans la bibliothèque d'API
4. Configurez l'écran de consentement OAuth (externe ou interne selon votre cas)
5. Créez des identifiants OAuth 2.0 pour une application Web
6. Ajoutez l'URI de redirection : `http://localhost:8000/api/google/callback`
7. Téléchargez le fichier JSON des identifiants
8. Renommez-le en `client_secret.json` et placez-le dans ce dossier

## Format du fichier client_secret.json

Le fichier doit avoir la structure suivante :

```json
{
  "web": {
    "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
    "project_id": "your-project-id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "YOUR_CLIENT_SECRET",
    "redirect_uris": ["http://localhost:8000/api/google/callback"]
  }
}
```

**IMPORTANT** : Ne partagez jamais votre fichier `client_secret.json` et ne le committez pas dans Git. Ce fichier contient des informations sensibles qui doivent rester privées.
