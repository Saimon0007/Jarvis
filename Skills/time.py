"""
Time skill for Jarvis: Responds with the current time and date.
"""

import datetime

def time_skill(user_input):
    now = datetime.datetime.now()
    return f"The current time is {now.strftime('%H:%M:%S')}"

def date_skill(user_input):
    today = datetime.date.today()
    return f"Today's date is {today.strftime('%Y-%m-%d')}"

def register(jarvis):
    jarvis.register_skill("time", time_skill)
    jarvis.register_skill("date", date_skill)
