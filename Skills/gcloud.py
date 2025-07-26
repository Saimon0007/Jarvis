
import os
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
from googleapiclient.discovery import build

def register(jarvis):
    def gcloud_translate(user_input):
        parts = user_input.split(maxsplit=2)
        if len(parts) < 3:
            return "Usage: gtranslate <target_language> <text>"
        target, text = parts[1], parts[2]
        try:
            client = translate.Client()
            result = client.translate(text, target_language=target)
            return f"Translated: {result['translatedText']}"
        except Exception as e:
            return f"Google Cloud error: {e}"

    def list_drive_files():
        """
        Lists the first 10 files in the user's Google Drive.
        Returns:
            list: List of file metadata dictionaries.
        """
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        scopes = ['https://www.googleapis.com/auth/drive.metadata.readonly']
        creds = service_account.Credentials.from_service_account_file(creds_path, scopes=scopes)
        service = build('drive', 'v3', credentials=creds)
        results = service.files().list(pageSize=10, fields="files(id, name)").execute()
        files = results.get('files', [])
        return files

    jarvis.register_skill('gtranslate', gcloud_translate)
    jarvis.register_skill('list_drive_files', list_drive_files)
