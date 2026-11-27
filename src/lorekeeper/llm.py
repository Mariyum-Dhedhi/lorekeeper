"""LLM factory. Both providers are free: Ollama runs locally, Groq has a free tier."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from lorekeeper.config import Provider, get_settings


def get_llm(provider: Provider | None = None) -> BaseChatModel:
    s = get_settings()
    provider = provider or s.llm_provider
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=s.ollama_model, base_url=s.ollama_base_url, temperature=0, num_ctx=8192
        )
    if provider == "groq":
        if not s.groq_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is empty. Get a free key at "
                "https://console.groq.com/keys and put it in .env"
            )
        from langchain_groq import ChatGroq

        return ChatGroq(model=s.groq_model, api_key=s.groq_api_key, temperature=0, max_retries=4)
    raise ValueError(f"Unknown provider: {provider}")


def get_judge_llm() -> BaseChatModel:
    s = get_settings()
    return get_llm(s.judge_provider or s.llm_provider)


def provider_label(provider: Provider | None = None) -> str:
    s = get_settings()
    provider = provider or s.llm_provider
    model = s.ollama_model if provider == "ollama" else s.groq_model
    return f"{provider}:{model}"


def chat(llm: BaseChatModel, system: str, user: str) -> str:
    """One-shot call returning plain text."""
    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = msg.content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content).strip()
