import os
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

def get_llm(api_key: str = None):
    """
    Attempts to get a ChatGoogleGenerativeAI model.
    Falls back to None if no API key is available.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
        
    try:
        # Using Gemini 1.5 Flash for fast agent response
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=key,
            temperature=0.2
        )
        return model
    except Exception as e:
        print(f"Error initializing ChatGoogleGenerativeAI: {str(e)}")
        return None

def add_agent_log(logs_list, agent_name, message):
    """
    Helper to append agent thought logs to the state.
    """
    logs_list.append({
        "agent": agent_name,
        "message": message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
