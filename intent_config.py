"""
Intent keywords configuration for Jarvis
Separated from main code for better maintainability
"""

INTENT_KEYWORDS = {
    "time": ["time", "clock", "current time", "what time"],
    "date": ["date", "today", "current date", "what date", "day"],
    "search": ["search", "find", "look up", "who is", "what is", "where is", "info", "information"],
    "gtranslate": ["translate", "gtranslate", "in spanish", "in french", "in german"],
    "emotion": ["emotion", "feel", "feeling", "mood", "sentiment"],
    "fetch": ["fetch", "download", "get url", "get web page", "web page"],
    "hello": ["hello", "hi", "hey", "greetings"],
    "jolly": ["joke", "cheer", "funny", "make me laugh", "jolly"],
    "nlp": ["nlp", "language", "intent", "meaning"],
    "summarize": ["summarize", "tl;dr", "shorten", "summary", "summarization"],
    "code": [
        # Programming languages
        "code", "generate code", "python", "java", "c++", "c#", "javascript", "typescript", 
        "go", "rust", "ruby", "php", "swift", "kotlin", "scala", "dart", "elixir", "f#", 
        "clojure", "julia", "perl", "objective-c", "vb.net", "delphi", "pascal", "groovy", 
        "abap", "cobol", "ada", "pl/sql", "sas", "stata", "verilog", "vhdl", "labview", 
        "scratch", "blockly", "apex", "solidity", "matlab", "r", "fortran", "assembly", 
        "prolog", "lisp", "haskell",
        
        # Code tasks
        "write a function", "write code", "program", "script", "explain code", "explain function", 
        "explain class", "how to code", "coding question", "debug", "fix code", "refactor", 
        "lint", "run code", "execute code", "compile", "algorithm", "data structure", "regex", 
        "sql", "bash", "shell", "powershell",
        
        # Web3 and blockchain
        "web3", "blockchain code", "smart contract", 
        
        # Machine Learning
        "ml code", "ai code", "deep learning code", "machine learning code", "pytorch", 
        "tensorflow", "keras", "scikit-learn", "opencv", "nlp code", "vision code", "audio code",
        
        # Game development
        "game code", "unity", "unreal", "godot", "shader", "glsl", "vulkan", "opengl", "directx",
        
        # Parallel computing
        "cuda", "opencl", "parallel code", "distributed code",
        
        # Cloud and DevOps
        "cloud code", "aws code", "azure code", "gcp code", "firebase code", "dockerfile", 
        "docker compose", "kubernetes yaml", "terraform", "ansible", "chef", "puppet", 
        "devops code", "ci/cd", "github actions", "gitlab ci", "bitbucket pipelines", 
        "travis ci", "circleci", "jenkins pipeline",
        
        # Testing
        "test code", "unit test", "integration test", "mock", "stub", "patch", "monkeypatch", 
        "property-based test", "fuzz test", "benchmark", "profile code",
        
        # Performance and security
        "performance code", "optimize code", "secure code", "cryptography code", "hash", 
        "encryption", "decryption", "jwt", "oauth", "sso",
        
        # APIs and networking
        "api code", "rest api", "graphql", "grpc", "websocket", "socket code", "network code", 
        "http code", "tcp code", "udp code",
        
        # Hardware and IoT
        "serial code", "usb code", "bluetooth code", "iot code", "robot code", "arduino code", 
        "raspberry pi code", "microcontroller code", "embedded code", "firmware code",
        
        # Hardware design
        "fpga code", "hardware code", "verilog code", "vhdl code", "hdl code"
    ]
}

# Multi-task routing instructions
MULTITASK_MAP = {
    "summarize": "Summarize the following:",
    "code": "Write code for the following:",
    "qa": "Answer the following question:",
}
