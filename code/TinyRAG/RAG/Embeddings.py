#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Optional, Tuple, Union
import torch
from transformers import AutoModel
from openai import OpenAI

os.environ['CURL_CA_BUNDLE'] = '' # 将 CURL_CA_BUNDLE 变量设置为空字符串，禁用 SSL 证书验证，即不使用任何证书文件进行验证。通常是在开发环境中，或者当你信任不安全的证书时使用。
_ = load_dotenv(find_dotenv())

class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path:str, is_api: bool) -> None:
        self.path =  path
        self.is_api = is_api

    def get_embedding(self, text: str, model: Optional[str] = "") -> List[float]:
        raise NotImplementedError # 抛出一个异常，告诉子类必须实现这个方法
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2:List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magintude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magintude:
            return 0
        return dot_product / magintude

class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """
    def __init__(self, path:str = '', is_api:bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY", "")
            self.client.base_url = os.getenv("OPENAI_BASE_URL", "")
    
    def get_embedding(self, text:str, model: str = "text-embedding-3-large") -> List[float]:
        if self.is_api:
            text = text.replace("\n", "" )
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding # type: ignore
        else:
            raise NotImplemented

class JinaEmbedding(BaseEmbeddings):
    """
    class for Jina embeddings
    """
    def __init__(self, path:str = 'jinaai/jina-embeddings-v2-base-zh', is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self._model = self.load_model()

    def get_embedding(self, text, model: Optional[str] = "") -> List[float]:
        """
        这里的实现逻辑应该是只能每次编码一个句子。
         即使不使用 model 参数，保留其一致性，确保接口统一。
        """
        return self._model.encode([text])[0].tolist()
    
    def load_model(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = AutoModel.from_pretrained(self.path, trust_remote_code=True).to(device)
        return model
    
class ZhipuEmbedding(BaseEmbeddings):
    def __init__(self, path: str, is_api: bool) -> None:
        super().__init__(path, is_api)