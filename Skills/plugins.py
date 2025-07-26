"""
Plugin management skill for Jarvis: List, enable, disable, and reload skills (plugins) at runtime.
Supports skill metadata, dependency checking, and status persistence.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Set
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

class SkillMetadata:
    """Store and manage skill metadata."""
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.dependencies: Set[str] = set()
        self.priority = 0
        self.description = ""
        self.version = "1.0.0"
        self.author = ""
        self.last_error: Optional[str] = None
        
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "dependencies": list(self.dependencies),
            "priority": self.priority,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "last_error": self.last_error
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'SkillMetadata':
        meta = cls(data["name"])
        meta.enabled = data.get("enabled", True)
        meta.dependencies = set(data.get("dependencies", []))
        meta.priority = data.get("priority", 0)
        meta.description = data.get("description", "")
        meta.version = data.get("version", "1.0.0")
        meta.author = data.get("author", "")
        meta.last_error = data.get("last_error")
        return meta

class SkillManager:
    """Manage skill metadata and dependencies."""
    def __init__(self):
        self.skills: Dict[str, SkillMetadata] = {}
        self.config_file = Path("skills/skill_config.json")
        self.load_config()
    
    def load_config(self) -> None:
        """Load skill configuration from file."""
        if self.config_file.exists():
            try:
                data = json.loads(self.config_file.read_text())
                self.skills = {
                    name: SkillMetadata.from_dict(meta)
                    for name, meta in data.items()
                }
            except Exception as e:
                logger.error(f"Failed to load skill config: {e}")
                self.skills = {}
    
    def save_config(self) -> None:
        """Save skill configuration to file."""
        try:
            data = {
                name: meta.to_dict()
                for name, meta in self.skills.items()
            }
            self.config_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save skill config: {e}")
    
    def get_skill_info(self, name: str) -> Optional[SkillMetadata]:
        """Get metadata for a skill."""
        return self.skills.get(name)
    
    def set_skill_status(self, name: str, enabled: bool, error: Optional[str] = None) -> None:
        """Update skill status and save config."""
        if name not in self.skills:
            self.skills[name] = SkillMetadata(name)
        self.skills[name].enabled = enabled
        self.skills[name].last_error = error
        self.save_config()
    
    def check_dependencies(self, name: str) -> List[str]:
        """Check if all dependencies for a skill are enabled."""
        meta = self.get_skill_info(name)
        if not meta:
            return []
        return [dep for dep in meta.dependencies 
                if dep not in self.skills or not self.skills[dep].enabled]

# Initialize skill manager
skill_manager = SkillManager()

def plugin_skill(user_input, conversation_history=None, search_skill=None):
    """
    Manage Jarvis skills (plugins): list, reload, enable, disable.
    Usage:
      plugins list
      plugins reload
      plugins disable <skill>
      plugins enable <skill>
    """
    import re
    import inspect
    from importlib import import_module, reload
    from pathlib import Path
    
    main_mod = import_module("main")
    jarvis_instance = getattr(main_mod, "jarvis", None)
    if not jarvis_instance:
        return "Jarvis instance not found."
        
    def format_skill_info(name: str, meta: Optional[SkillMetadata] = None) -> str:
        """Format skill information for display."""
        if not meta:
            meta = skill_manager.get_skill_info(name) or SkillMetadata(name)
        
        status = "✓ Enabled" if meta.enabled else "✗ Disabled"
        if meta.last_error:
            status += f" (Error: {meta.last_error})"
            
        info = [
            f"**{name}**",
            f"Status: {status}",
            f"Version: {meta.version}",
        ]
        
        if meta.description:
            info.append(f"Description: {meta.description}")
        if meta.author:
            info.append(f"Author: {meta.author}")
        if meta.dependencies:
            info.append(f"Dependencies: {', '.join(meta.dependencies)}")
            
        return "\n".join(info)

    def validate_skill_name(skill: str) -> bool:
        """Check if skill name is valid and exists."""
        if not re.match(r'^[a-zA-Z0-9_]+$', skill):
            return False
        return Path(f"skills/{skill}.py").exists()

    def update_skill_metadata(module, skill_name: str) -> None:
        """Extract and update skill metadata from module."""
        meta = skill_manager.get_skill_info(skill_name) or SkillMetadata(skill_name)
        
        # Get metadata from module docstring
        if module.__doc__:
            meta.description = module.__doc__.split('\n')[0]
        
        # Extract version and author from module attributes
        meta.version = getattr(module, '__version__', meta.version)
        meta.author = getattr(module, '__author__', meta.author)
        
        # Extract dependencies from imports
        try:
            source = inspect.getsource(module)
            imports = re.findall(r'from skills\.([\w]+) import', source)
            meta.dependencies = set(imports)
        except Exception:
            pass
        
        skill_manager.skills[skill_name] = meta
        skill_manager.save_config()

    cmd_parts = user_input[len("plugins"):].strip().split()
    if not cmd_parts:
        return """Available commands:
- plugins list [--all]: List enabled or all skills
- plugins info <skill>: Show detailed skill information
- plugins reload [<skill>]: Reload specific or all skills
- plugins disable <skill>: Disable a skill
- plugins enable <skill>: Enable a skill
- plugins cleanup: Remove references to missing skills"""

    cmd = cmd_parts[0].lower()
    args = cmd_parts[1:] if len(cmd_parts) > 1 else []

    if cmd == "list":
        show_all = len(args) > 0 and args[0] == "--all"
        skills_info = []
        
        for name, skill in jarvis_instance.skills.items():
            meta = skill_manager.get_skill_info(name)
            if show_all or (meta and meta.enabled):
                skills_info.append(format_skill_info(name, meta))
                
        if not skills_info:
            return "No skills found."
        return "\n\n".join(skills_info)

    elif cmd == "info":
        if not args:
            return "Please specify a skill name."
        skill = args[0]
        if not validate_skill_name(skill):
            return f"Invalid skill name: {skill}"
            
        try:
            module = import_module(f"skills.{skill}")
            update_skill_metadata(module, skill)
            return format_skill_info(skill)
        except Exception as e:
            return f"Error getting skill info: {e}"

    elif cmd == "reload":
        if args:
            skill = args[0]
            if not validate_skill_name(skill):
                return f"Invalid skill name: {skill}"
            try:
                module = import_module(f"skills.{skill}")
                reload(module)
                if hasattr(module, "register"):
                    module.register(jarvis_instance)
                    update_skill_metadata(module, skill)
                    return f"Skill '{skill}' reloaded successfully."
                return f"Skill '{skill}' does not have a register() function."
            except Exception as e:
                error_msg = str(e)
                skill_manager.set_skill_status(skill, False, error_msg)
                return f"Failed to reload skill '{skill}': {error_msg}"
        else:
            try:
                jarvis_instance.reload_skills()
                return "All skills reloaded successfully."
            except Exception as e:
                return f"Error reloading skills: {e}"

    elif cmd == "disable":
        if not args:
            return "Please specify a skill name."
        skill = args[0]
        if not validate_skill_name(skill):
            return f"Invalid skill name: {skill}"
            
        # Check if other skills depend on this one
        dependents = []
        for name, meta in skill_manager.skills.items():
            if skill in meta.dependencies and meta.enabled:
                dependents.append(name)
                
        if dependents:
            return (f"Cannot disable '{skill}' because these skills depend on it: "
                   f"{', '.join(dependents)}")
                   
        result = jarvis_instance.unload_skill(skill)
        skill_manager.set_skill_status(skill, False)
        return result

    elif cmd == "enable":
        if not args:
            return "Please specify a skill name."
        skill = args[0]
        if not validate_skill_name(skill):
            return f"Invalid skill name: {skill}"
            
        # Check dependencies
        missing_deps = skill_manager.check_dependencies(skill)
        if missing_deps:
            return (f"Cannot enable '{skill}' because these dependencies are not enabled: "
                   f"{', '.join(missing_deps)}")
                   
        try:
            module = import_module(f"skills.{skill}")
            if hasattr(module, "register"):
                module.register(jarvis_instance)
                update_skill_metadata(module, skill)
                skill_manager.set_skill_status(skill, True, None)
                return f"Skill '{skill}' enabled successfully."
            else:
                return f"Skill '{skill}' does not have a register() function."
        except Exception as e:
            error_msg = str(e)
            skill_manager.set_skill_status(skill, False, error_msg)
            return f"Failed to enable skill '{skill}': {error_msg}"

    elif cmd == "cleanup":
        removed = []
        for name in list(skill_manager.skills.keys()):
            if not Path(f"skills/{name}.py").exists():
                removed.append(name)
                del skill_manager.skills[name]
        skill_manager.save_config()
        if removed:
            return f"Removed {len(removed)} missing skills: {', '.join(removed)}"
        return "No missing skills found."

    else:
        return """Available commands:
- plugins list [--all]: List enabled or all skills
- plugins info <skill>: Show detailed skill information
- plugins reload [<skill>]: Reload specific or all skills
- plugins disable <skill>: Disable a skill
- plugins enable <skill>: Enable a skill
- plugins cleanup: Remove references to missing skills"""

def register(jarvis):
    jarvis.register_skill("plugins", plugin_skill)
