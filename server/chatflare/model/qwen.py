from chatflare.model.base import ModelBase  # 继承基础模型类
from typing import Optional, Dict, Any
import requests
import aiohttp
import os

class QwenLLM(ModelBase):  # 继承ModelBase
    def __init__(
        self,
        model_name: str = "qwen3-8B",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name,** kwargs)  # 调用父类初始化
        self.api_key = api_key or os.getenv("QWEN_API_KEY")
        self.base_url = base_url or "https://api.qwen.com/v1/chat/completions"
        self.temperature = temperature
        self.max_tokens = max_tokens
        if not self.api_key:
            raise ValueError("QWEN_API_KEY must be provided")

    def _create_payload(self, prompt: str, json_mode: bool = False) -> Dict[str, Any]:
        """构建请求参数（抽取为通用方法）"""
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def predict(self, prompt: str, json_mode: bool = False) -> str:
        """同步生成方法（实现ModelBase的接口）"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = self._create_payload(prompt, json_mode)
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=60  # 增加超时控制
        )
        response.raise_for_status()  # 抛出HTTP错误
        return response.json()["choices"][0]["message"]["content"]

    async def apredict(self, prompt: str, json_mode: bool = False) -> str:
        """异步生成方法（实现ModelBase的接口）"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = self._create_payload(prompt, json_mode)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()
                return result["choices"][0]["message"]["content"]