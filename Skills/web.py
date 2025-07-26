import requests

def register(jarvis):
    def fetch_url(user_input):
        parts = user_input.split(maxsplit=1)
        if len(parts) < 2:
            help_text = (
                "Usage: fetch <url>\n"
                "Try these resources to learn advanced cloud/API integrations:\n"
                "- https://realpython.com/api-integration-in-python/\n"
                "- https://www.freecodecamp.org/news/how-to-build-an-api-in-python/\n"
                "- https://www.fullstackpython.com/cloud-computing.html\n"
            )
            return help_text
        url = parts[1]
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            # Return first 300 characters for brevity
            return response.text[:300] + ("..." if len(response.text) > 300 else "")
        except Exception as e:
            return f"Error fetching URL: {e}"
    jarvis.register_skill('fetch', fetch_url)
