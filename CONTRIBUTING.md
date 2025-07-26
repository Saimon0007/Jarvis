# Contributing to Jarvis AI Assistant

## 👋 Welcome!
Thank you for your interest in contributing to Jarvis! This document provides guidelines and instructions for contributing to the project. Whether you're fixing bugs, adding new skills, or improving documentation, your help is welcome and appreciated.

## 🗂️ Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Guidelines](#development-guidelines)
- [Creating Skills](#creating-skills)
- [Pull Request Process](#pull-request-process)
- [Bug Reports and Feature Requests](#bug-reports-and-feature-requests)
- [Style Guidelines](#style-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)

## 📜 Code of Conduct
By participating in this project, you agree to maintain a respectful and inclusive environment. We expect all contributors to:
- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what's best for the community
- Show empathy towards other community members

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic understanding of AI/ML concepts
- Familiarity with Flask (for web interface contributions)

### Setup Development Environment
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Jarvis.git
   cd Jarvis
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Create a new branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 💻 Development Guidelines

### General Principles
1. **Modularity**: Keep code modular and focused
2. **Documentation**: Document all functions, classes, and complex logic
3. **Testing**: Include tests for new features
4. **Error Handling**: Implement proper error handling and logging
5. **Performance**: Consider performance implications of changes

### Code Organization
```
jarvis/
├── skills/                 # Individual skill modules
├── static/                 # Web UI static files
├── tests/                 # Test files
├── main.py                # Core Jarvis functionality
├── api_server.py          # Web API implementation
└── intent_config.py       # Intent configuration
```

## 🛠️ Creating Skills

### Skill Module Template
```python
"""
Skill Name: Description of what the skill does
Usage: Example of how to use the skill
"""

def skill_function(user_input, **kwargs):
    """
    Main skill function
    
    Args:
        user_input (str): User's input text
        **kwargs: Additional context parameters
    
    Returns:
        str: Response to user
    """
    # Skill implementation
    pass

def register(jarvis):
    """Register the skill with Jarvis."""
    jarvis.register_skill("skill_name", skill_function)
```

### Skill Guidelines
1. Each skill should have a single responsibility
2. Include proper error handling
3. Document usage and parameters
4. Consider context and conversation history
5. Implement proper input validation
6. Return clear, helpful responses

## 📝 Pull Request Process

1. **Before Submitting**
   - Update documentation if adding new features
   - Add/update tests as needed
   - Run the test suite locally
   - Update requirements.txt if adding dependencies

2. **Submission Process**
   - Create a descriptive PR title
   - Fill out the PR template
   - Link related issues
   - Request review from maintainers

3. **Review Process**
   - Address review comments
   - Keep the PR focused and scoped
   - Be responsive to feedback
   - Update your PR as needed

## 🐛 Bug Reports and Feature Requests

### Bug Reports
Include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- System information
- Error messages/logs

### Feature Requests
Include:
- Clear description of the feature
- Use cases and benefits
- Potential implementation approach
- Considerations and challenges

## 📋 Style Guidelines

### Python Code Style
- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Keep functions focused and concise
- Use type hints where appropriate
- Include docstrings for all public functions/classes

### Commit Messages
Format:
```
type(scope): Brief description

Detailed description of changes
```

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Code style changes
- refactor: Code refactoring
- test: Adding tests
- chore: Maintenance tasks

## ✅ Testing Guidelines

### Test Requirements
- Write unit tests for new features
- Include integration tests where appropriate
- Maintain test coverage above 80%
- Test error cases and edge conditions

### Test Structure
```python
def test_feature():
    """
    Test description
    
    Arrange: Setup test conditions
    Act: Execute the feature
    Assert: Verify the results
    """
    # Test implementation
```

## 📚 Documentation Guidelines

### Code Documentation
- Use clear and concise docstrings
- Document parameters and return values
- Include usage examples
- Explain complex algorithms

### User Documentation
- Keep README.md updated
- Document new features
- Include usage examples
- Update change log

## 🤝 Community

### Getting Help
- Join our Discord server
- Check existing issues
- Ask questions in discussions
- Read the documentation

### Recognition
Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in documentation

---

Last Updated: 2025-07-26
