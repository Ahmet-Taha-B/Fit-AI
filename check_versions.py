try:
    import langgraph
    print(f"langgraph: {langgraph.__version__}")
except ImportError:
    print("langgraph: Not installed")
except AttributeError:
    print("langgraph: No __version__ attribute")

try:
    import chromadb
    print(f"chromadb: {chromadb.__version__}")
except ImportError:
    print("chromadb: Not installed")
except AttributeError:
    print("chromadb: No __version__ attribute")

try:
    import langchain_groq
    print(f"langchain_groq: {langchain_groq.__version__}")
except ImportError:
    print("langchain_groq: Not installed")
except AttributeError:
    print("langchain_groq: No __version__ attribute")
