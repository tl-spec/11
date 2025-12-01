import os
from dotenv import load_dotenv

def load_environment_variables():
    if os.getenv('OPENAI_API_KEY') is None:
        load_dotenv('/app/.env')
        load_dotenv('/app/.env.config')
        load_dotenv('.env')
        load_dotenv('.env.config')
        load_dotenv('server/.env.config')
        print("\n=== Current Environment Settings ===")
        print(f"Qwen API Key:{'SET' if os.getenv('QWEN_API_KEY') else 'Not Set'}")
        print(f"OpenAI API Key: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not Set'}")
        if os.getenv('OPENAI_API_KEY') is None:
            print("Warning: The OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        print(f"LLM Model Name: {os.getenv('LLM_MODEL_NAME', 'Not Set')}")
        print(f"LLM Temperature: {os.getenv('LLM_TEMPERATURE', 'Not Set')}")
        print(f"LLM Max Tokens: {os.getenv('LLM_MAX_TOKENS', 'Not Set')}")
        print("==================================\n")
        if 'qwen' in os.getenv('LLM_MODEL_NAME', '').lower():
            if not os.getenv('QWEN_API_KEY'):
                print("Warning: Qwen is not supported yet. Please use OpenAI API.")
                exit(1)
        elif 'gpt' in os.getenv('LLM_MODEL_NAME', '').lower():
            if not os.getenv('OPENAI_API_KEY'):
                print("Warning: OpenAI is not supported yet. Please use Qwen API.")
                exit(1)
        if os.getenv('OPENAI_API_KEY') is None:
            print("Warning: The OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
            print("Please set the OPENAI_API_KEY environment variable.")
            exit(1)
