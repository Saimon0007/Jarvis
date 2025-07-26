"""
Weather skill for Jarvis: Get current weather and forecast.
Extend this module to use real weather APIs (e.g., OpenWeatherMap).
"""
import datetime

def weather_skill(user_input, conversation_history=None):
    """
    Show current weather and today's forecast. (Demo version)
    Usage: 'weather', 'what's the weather', 'weather forecast'
    """
    try:
        # Placeholder: In production, fetch from a weather API
        today = datetime.date.today().strftime('%A, %B %d, %Y')
        location = "New York"
        current = "Sunny, 25°C"
        forecast = [
            {'time': 'Morning', 'desc': 'Sunny, 22°C'},
            {'time': 'Afternoon', 'desc': 'Partly cloudy, 25°C'},
            {'time': 'Evening', 'desc': 'Clear, 20°C'},
        ]
        forecast_lines = [f"- {f['time']}: {f['desc']}" for f in forecast]
        return f"Weather for {location} on {today}:\nCurrent: {current}\nForecast:\n" + "\n".join(forecast_lines)
    except Exception as e:
        return f"Sorry, I couldn't fetch the weather. ({e})"

def register(jarvis):
    jarvis.register_skill('weather', weather_skill)
