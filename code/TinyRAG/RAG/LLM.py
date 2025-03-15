#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Dict, List, Optional, Tuple, Literal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPALTE="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:""",
    InternLM_PROMPT_TEMPALTE="""先对上下文进行内容总结,再使用上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
)

class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path
    
    def chat(self, prompt: str, history: List[dict], content: str) -> None:
        pass

    def load_model(self):
        pass

class OpenAIChat(BaseModel):
    def __init__(self, path: str = '', model: str = "gpt-3.5-turbo-1106") -> None:
        super().__init__(path)
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY", "")
        client.base_url = os.getenv("OPENAI_BASE_URL", "")
        history.append({'role': 'user', 'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, content=content)})
        response = client.chat.completions.create(
            model=self.model,
            messages=history, # type: ignore
            max_tokens=150,
            temperature=0.1
        ) 
        return response.choices[0].message.content or "无返回结果"
    
class InternLMChat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16, trust_remote_code=True).to(device)

    def chat(self, prompt: str, history: List = [], content: str = ''):
        prompt = PROMPT_TEMPLATE['InternLM_PROMPT_TEMPALTE'].format(question=prompt, content=content)
        response, history = self.model.chat(self.tokenizer, prompt, history)
        return response
