"""
Microsoft Graph skill module for Jarvis.
Provides functions to interact with Microsoft APIs (e.g., Microsoft Graph).
"""
import os
import msal

def get_graph_token():
    """
    Acquires an access token for Microsoft Graph API using client credentials.
    Returns:
        str: Access token or error message.
    """
    client_id = os.getenv("MICROSOFT_CLIENT_ID")
    client_secret = os.getenv("MICROSOFT_CLIENT_SECRET")
    tenant_id = os.getenv("MICROSOFT_TENANT_ID")
    authority = f"https://login.microsoftonline.com/{tenant_id}"
    scope = ["https://graph.microsoft.com/.default"]

    app = msal.ConfidentialClientApplication(
        client_id, authority=authority, client_credential=client_secret
    )
    result = app.acquire_token_for_client(scopes=scope)
    if "access_token" in result:
        return result["access_token"]
    else:
        return f"Error acquiring token: {result.get('error_description', result)}"

def register(jarvis):
    """
    Register the Microsoft Graph skill with Jarvis.
    """
    jarvis.register_skill("get_graph_token", get_graph_token)
