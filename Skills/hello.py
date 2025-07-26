def register(jarvis):
    def hello_skill(user_input):
        return "Hello! How can I assist you today?"
    jarvis.register_skill('hello', hello_skill)
