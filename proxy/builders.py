"""Builders for docling-serve VLM config payloads."""
import json
from proxy.config import (
    DEFAULT_VLM_URL, DEFAULT_VLM_API_KEY, DEFAULT_VLM_MODEL,
    DEFAULT_VLM_TIMEOUT, DEFAULT_VLM_CONCURRENCY,
    DEFAULT_VLM_MAX_COMPLETION_TOKENS, DEFAULT_VLM_SCALE,
    VLM_PROXY_ENABLED, VLM_PROXY_URL,
)
from proxy.prompts import DEFAULT_VLM_PROMPT, DEFAULT_VLM_PIPELINE_PROMPT


def _vlm_proxy_url(profile: str) -> str:
    """URL для инжекции в конфиги docling-serve."""
    if VLM_PROXY_ENABLED and VLM_PROXY_URL:
        sep = "&" if "?" in VLM_PROXY_URL else "?"
        return f"{VLM_PROXY_URL}{sep}profile={profile}"
    return DEFAULT_VLM_URL


def build_picture_description_api(vlm_overrides: dict) -> str:
    params = {"model": vlm_overrides.get("vlm_model", DEFAULT_VLM_MODEL), "chat_template_kwargs": {"enable_thinking": False}}
    if "vlm_temperature" in vlm_overrides:
        params["temperature"] = float(vlm_overrides["vlm_temperature"])
    if "vlm_max_tokens" in vlm_overrides:
        params["max_tokens"] = int(vlm_overrides["vlm_max_tokens"])
    api_config = {
        "url": vlm_overrides.get("vlm_url", _vlm_proxy_url("picture_desc")),
        "headers": {"Authorization": f"Bearer {vlm_overrides.get('vlm_api_key', DEFAULT_VLM_API_KEY)}"},
        "params": params,
        "timeout": int(vlm_overrides.get("vlm_timeout", DEFAULT_VLM_TIMEOUT)),
        "concurrency": int(vlm_overrides.get("vlm_concurrency", DEFAULT_VLM_CONCURRENCY)),
        "prompt": vlm_overrides.get("vlm_prompt", DEFAULT_VLM_PROMPT) + "\n/no_think"
    }
    return json.dumps(api_config)


def build_custom_model(vlm_overrides: dict = {}, classification: str = "false") -> str:
    api_config = {
        "engine_options": {
            "engine_type": "api_openai",
            "url": DEFAULT_VLM_URL,
            "headers": {"Authorization": f"Bearer {DEFAULT_VLM_API_KEY}"},
            "timeout": 300
        },
        "model_spec": {
            "name": "Qwen3-VL",
            "default_repo_id": "Qwen/Qwen3-VL-32B-Instruct",
            "prompt": DEFAULT_VLM_PROMPT + "\n/no_think",
            "response_format": "markdown",
            "api_overrides": {
                "api_openai": {
                    "params": {
                        "model": vlm_overrides.get("vlm_model", DEFAULT_VLM_MODEL),
                        "max_completion_tokens": int(vlm_overrides.get("vlm_max_completion_tokens", DEFAULT_VLM_MAX_COMPLETION_TOKENS)),
                        "chat_template_kwargs": {"enable_thinking": False}
                    }
                }
            }
        },
        "prompt": DEFAULT_VLM_PROMPT + "\n/no_think",
        "batch_size": 1,
        "concurrency": int(vlm_overrides.get("vlm_concurrency", DEFAULT_VLM_CONCURRENCY)),
        "scale": float(vlm_overrides.get("vlm_scale", DEFAULT_VLM_SCALE)),
        "picture_area_threshold": 0.01,
        "generation_config": {"max_new_tokens": 2048, "do_sample": False}
    }

    if classification == "true":
        api_config["classification_min_confidence"] = 0.8
        api_config["classification_deny"] = ['icon', 'logo', 'signature', 'stamp', 'qr_code', 'bar_code']

    return json.dumps(api_config)


def build_vlm_pipeline_model_api(vlm_overrides: dict = {}) -> str:
    """VlmModelApi flat format for vlm_pipeline_model_api.

    Sampling-параметры теперь инжектируются на уровне VLM proxy.
    """
    config = {
        "url": vlm_overrides.get("vlm_url", _vlm_proxy_url("full_page")),
        "headers": {"Authorization": f"Bearer {vlm_overrides.get('vlm_api_key', DEFAULT_VLM_API_KEY)}"},
        "params": {
            "model": vlm_overrides.get("vlm_model", DEFAULT_VLM_MODEL),
            "max_completion_tokens": int(vlm_overrides.get("vlm_max_completion_tokens", DEFAULT_VLM_MAX_COMPLETION_TOKENS)),
            "chat_template_kwargs": {"enable_thinking": False}
        },
        "prompt": vlm_overrides.get("vlm_pipeline_prompt", DEFAULT_VLM_PIPELINE_PROMPT) + "\n/no_think",
        "response_format": "markdown",
        "timeout": int(vlm_overrides.get("vlm_timeout", DEFAULT_VLM_TIMEOUT)),
        "concurrency": int(vlm_overrides.get("vlm_concurrency", DEFAULT_VLM_CONCURRENCY)),
        "scale": float(vlm_overrides.get("vlm_scale", DEFAULT_VLM_SCALE)),
    }
    return json.dumps(config)
