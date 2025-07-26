"""
System Monitor skill for Jarvis: Show CPU, RAM, and disk usage.
Extend this module for more detailed system stats.
"""
import psutil

def sysmonitor_skill(user_input, conversation_history=None):
    """
    Show system resource usage. Usage: 'system status', 'cpu usage', 'memory usage', 'disk usage'
    """
    try:
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        return (
            f"System Status:\n"
            f"- CPU Usage: {cpu}%\n"
            f"- Memory Usage: {mem.percent}% ({mem.used // (1024**2)}MB/{mem.total // (1024**2)}MB)\n"
            f"- Disk Usage: {disk.percent}% ({disk.used // (1024**3)}GB/{disk.total // (1024**3)}GB)"
        )
    except Exception as e:
        return f"Sorry, I couldn't get system status. ({e})"

def register(jarvis):
    jarvis.register_skill('sysmonitor', sysmonitor_skill)
