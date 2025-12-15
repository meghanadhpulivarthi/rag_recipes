import os
import requests
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_rits_model_list():
    url = "https://rits.fmaas.res.ibm.com/ritsapi/inferenceinfo"
    response = requests.get(url, headers={"RITS_API_KEY": os.getenv("RITS_API_KEY")})
    if response.status_code == 200:
        return {m["model_name"]: m["endpoint"] for m in response.json()}
    raise Exception(f"Failed fetching RITS model list:\n\n{response.text}")


def get_rits_llm(model_name: str):
    """Initialize ChatOpenAI client for RITS model."""
    minfo = get_rits_model_list()
    if model_name not in minfo:
        raise ValueError(f"Model '{model_name}' not found in RITS registry.")
    url = f"{minfo[model_name]}/v1"
    print(f"🌐 Using RITS model '{model_name}' ({url})")
    return ChatOpenAI(
        model=model_name,
        max_retries=2,
        api_key="/",
        base_url=url,
        default_headers={"RITS_API_KEY": os.getenv("RITS_API_KEY")},
    )
