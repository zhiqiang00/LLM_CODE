# -*- coding: utf-8 -*-
import os
import numpy as np
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Tuple, Union

os.environ['CURL_CA_BUNDLE'] = '' # 将 CURL_CA_BUNDLE 变量设置为空字符串，禁用 SSL 证书验证，即不使用任何证书文件进行验证。通常是在开发环境中，或者当你信任不安全的证书时使用。
_ = load_dotenv(find_dotenv())

print(os.environ.get("OPENAI_API_KEY")) # 获取环境变量 OPENAI_API_KEY 的值

class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path:str, is_api: bool) -> None:
        self.path =  path
        self.is_api = is_api

    def get_embedding(self, text: str, model: str) -> List[float]:
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

        