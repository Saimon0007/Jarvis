// ========================================
// JARVIS AI Assistant - UI JavaScript
// ========================================

class JarvisUI {
    constructor() {
        this.isListening = false;
        this.recognition = null;
        this.currentUser = 'You';
        this.apiBase = '/api';
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupSpeechRecognition();
        this.loadSkills();
        this.updateCharacterCount();
    }

    setupEventListeners() {
        // Message input
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        messageInput.addEventListener('input', () => {
            this.updateCharacterCount();
            this.autoResize(messageInput);
        });

        sendBtn.addEventListener('click', () => this.sendMessage());

        // Voice button
        document.getElementById('voiceBtn').addEventListener('click', () => {
            this.toggleVoiceRecognition();
        });

        // Close modals on outside click
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal-overlay')) {
                this.closeAllModals();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === '/') {
                e.preventDefault();
                messageInput.focus();
            }
            if (e.key === 'Escape') {
                this.closeAllModals();
                this.closeFabMenu();
                this.closeSkillsPanel();
            }
        });
    }

    setupSpeechRecognition() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.recognition = new SpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = 'en-US';

            this.recognition.onstart = () => {
                this.isListening = true;
                this.updateVoiceButton();
            };

            this.recognition.onend = () => {
                this.isListening = false;
                this.updateVoiceButton();
            };

            this.recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                document.getElementById('messageInput').value = transcript;
                this.updateCharacterCount();
            };

            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.showNotification('Voice recognition error: ' + event.error, 'error');
            };
        }
    }

    toggleVoiceRecognition() {
        if (!this.recognition) {
            this.showNotification('Speech recognition not supported in this browser', 'warning');
            return;
        }

        if (this.isListening) {
            this.recognition.stop();
        } else {
            this.recognition.start();
        }
    }

    updateVoiceButton() {
        const voiceBtn = document.getElementById('voiceBtn');
        const icon = voiceBtn.querySelector('i');
        
        if (this.isListening) {
            icon.className = 'fas fa-stop';
            voiceBtn.classList.add('listening');
        } else {
            icon.className = 'fas fa-microphone';
            voiceBtn.classList.remove('listening');
        }
    }

    autoResize(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }

    updateCharacterCount() {
        const messageInput = document.getElementById('messageInput');
        const charCount = document.getElementById('charCount');
        const current = messageInput.value.length;
        const max = messageInput.maxLength;
        
        charCount.textContent = `${current} / ${max}`;
        
        if (current > max * 0.9) {
            charCount.style.color = 'var(--warning-color)';
        } else if (current === max) {
            charCount.style.color = 'var(--error-color)';
        } else {
            charCount.style.color = 'var(--text-muted)';
        }
    }

    async sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        
        if (!message) return;

        // Hide suggestions panel
        const suggestionsPanel = document.getElementById('suggestionsPanel');
        if (suggestionsPanel) {
            suggestionsPanel.style.display = 'none';
        }

        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input
        messageInput.value = '';
        this.updateCharacterCount();
        messageInput.style.height = 'auto';

        // Show loading
        this.showLoading();

        try {
            // Send to backend
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    user_id: this.currentUser
                })
            });

            const data = await response.json();
            
            if (data.response) {
                this.addMessage(data.response, 'jarvis');
            } else {
                throw new Error('No response from server');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('Sorry, I encountered an error. Please try again.', 'jarvis');
        } finally {
            this.hideLoading();
        }
    }

    addMessage(text, sender) {
        const chatMessages = document.getElementById('chatMessages');
        const messageGroup = document.createElement('div');
        messageGroup.className = `message-group ${sender}-message`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble';

        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        messageText.textContent = text;

        const timestamp = document.createElement('div');
        timestamp.className = 'message-timestamp';
        timestamp.textContent = new Date().toLocaleTimeString();

        messageBubble.appendChild(messageText);
        messageBubble.appendChild(timestamp);
        messageContent.appendChild(messageBubble);
        messageGroup.appendChild(avatar);
        messageGroup.appendChild(messageContent);

        chatMessages.appendChild(messageGroup);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Add typing effect for Jarvis messages
        if (sender === 'jarvis') {
            this.addTypingEffect(messageText);
        }
    }

    addTypingEffect(element) {
        const text = element.textContent;
        element.textContent = '';
        element.classList.add('typing-effect');
        
        let i = 0;
        const typeWriter = () => {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(typeWriter, 30);
            } else {
                element.classList.remove('typing-effect');
            }
        };
        
        setTimeout(typeWriter, 500);
    }

    showLoading() {
        document.getElementById('loadingOverlay').classList.add('active');
    }

    hideLoading() {
        document.getElementById('loadingOverlay').classList.remove('active');
    }

    async loadSkills() {
        try {
            const response = await fetch('/skills');
            const data = await response.json();
            
            if (data.skills) {
                this.displaySkills(data.skills);
            }
        } catch (error) {
            console.error('Error loading skills:', error);
            this.displaySkills(['Error loading skills']);
        }
    }

    displaySkills(skills) {
        const skillsList = document.getElementById('skillsList');
        skillsList.innerHTML = '';
        
        skills.forEach(skill => {
            const skillItem = document.createElement('div');
            skillItem.className = 'skill-item';
            skillItem.innerHTML = `
                <i class="fas fa-cog"></i>
                <span>${skill}</span>
            `;
            skillItem.addEventListener('click', () => {
                this.sendSkillCommand(skill);
            });
            skillsList.appendChild(skillItem);
        });
    }

    sendSkillCommand(skill) {
        const messageInput = document.getElementById('messageInput');
        messageInput.value = skill;
        this.closeSkillsPanel();
        messageInput.focus();
    }

    // UI Controls
    toggleFabMenu() {
        const fabContainer = document.querySelector('.fab-container');
        fabContainer.classList.toggle('active');
    }

    closeFabMenu() {
        const fabContainer = document.querySelector('.fab-container');
        fabContainer.classList.remove('active');
    }

    toggleSkillsPanel() {
        const skillsPanel = document.getElementById('skillsPanel');
        skillsPanel.classList.toggle('active');
        this.closeFabMenu();
    }

    closeSkillsPanel() {
        const skillsPanel = document.getElementById('skillsPanel');
        skillsPanel.classList.remove('active');
    }

    openSettings() {
        document.getElementById('settingsModal').classList.add('active');
        this.closeFabMenu();
    }

    closeSettings() {
        document.getElementById('settingsModal').classList.remove('active');
    }

    clearChat() {
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = `
            <div class="message-group jarvis-message">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="message-bubble">
                        <div class="message-text">
                            <div class="typing-effect">
                                Chat cleared! How can I help you today?
                            </div>
                        </div>
                        <div class="message-timestamp">Just now</div>
                    </div>
                </div>
            </div>
        `;
        
        // Show suggestions panel again
        const suggestionsPanel = document.getElementById('suggestionsPanel');
        if (suggestionsPanel) {
            suggestionsPanel.style.display = 'block';
        }
        
        this.closeFabMenu();
    }

    closeAllModals() {
        document.querySelectorAll('.modal-overlay').forEach(modal => {
            modal.classList.remove('active');
        });
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        // Style the notification
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '16px 24px',
            borderRadius: '8px',
            color: 'white',
            fontWeight: '500',
            zIndex: '10000',
            opacity: '0',
            transform: 'translateX(100%)',
            transition: 'all 0.3s ease'
        });

        // Set background color based on type
        const colors = {
            info: 'var(--primary-color)',
            success: 'var(--success-color)',
            warning: 'var(--warning-color)',
            error: 'var(--error-color)'
        };
        notification.style.background = colors[type] || colors.info;

        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.style.opacity = '1';
            notification.style.transform = 'translateX(0)';
        }, 100);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
}

// Global functions for HTML onclick events
function sendSuggestion(suggestion) {
    const messageInput = document.getElementById('messageInput');
    messageInput.value = suggestion;
    jarvisUI.sendMessage();
}

function toggleFabMenu() {
    jarvisUI.toggleFabMenu();
}

function toggleSkillsPanel() {
    jarvisUI.toggleSkillsPanel();
}

function openSettings() {
    jarvisUI.openSettings();
}

function closeSettings() {
    jarvisUI.closeSettings();
}

function clearChat() {
    jarvisUI.clearChat();
}

// Initialize when DOM is loaded
let jarvisUI;
document.addEventListener('DOMContentLoaded', () => {
    jarvisUI = new JarvisUI();
    
    // Add some startup effects
    setTimeout(() => {
        const typingElements = document.querySelectorAll('.typing-effect');
        typingElements.forEach(element => {
            element.style.animationPlayState = 'running';
        });
    }, 500);
});

// Add CSS for voice button listening state
const style = document.createElement('style');
style.textContent = `
    .voice-btn.listening {
        background: var(--error-color) !important;
        color: white !important;
        border-color: var(--error-color) !important;
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .notification {
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
`;
document.head.appendChild(style);
