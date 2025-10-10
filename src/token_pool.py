#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Token池管理模块
"""

import os
from typing import List, Optional
from pathlib import Path
from .helpers import debug_log


class TokenPool:
    """Token池管理类"""
    
    def __init__(self):
        """初始化Token池"""
        self.tokens: List[str] = []
        self.current_index: int = 0
        self.current_token: Optional[str] = None
        self._load_tokens()
    
    def _load_tokens(self):
        """从.env和tokens.txt加载token并去重"""
        token_set = set()
        
        # 1. 从环境变量ZAI_TOKEN加载（必须存在）
        zai_token = os.getenv("ZAI_TOKEN", "").strip()
        if not zai_token:
            raise ValueError("[ERROR] 未配置ZAI_TOKEN，请在.env文件中设置")
        
        # 处理多个token（逗号分割）
        env_tokens = [token.strip() for token in zai_token.split(",") if token.strip()]
        token_set.update(env_tokens)
        debug_log(f"从环境变量ZAI_TOKEN加载了 {len(env_tokens)} 个token")
        
        # 2. 从tokens.txt加载（可选）
        tokens_file = Path("tokens.txt")
        if tokens_file.exists():
            try:
                with open(tokens_file, 'r', encoding='utf-8') as f:
                    file_tokens = [line.strip() for line in f if line.strip()]
                    file_tokens_count = len(file_tokens)
                    token_set.update(file_tokens)
                    debug_log(f"从tokens.txt加载了 {file_tokens_count} 个token")
            except Exception as e:
                debug_log(f"[WARN] 读取tokens.txt失败: {e}")
        else:
            debug_log("tokens.txt文件不存在，跳过加载")
        
        # 去重后的token列表
        self.tokens = list(token_set)
        
        if not self.tokens:
            raise ValueError("[ERROR] 没有可用的token")
        
        # 初始化当前token
        self.current_token = self.tokens[0]
        self.current_index = 0
        
        debug_log(f"[OK] Token池初始化完成，共 {len(self.tokens)} 个唯一token")
    
    def get_token(self) -> str:
        """获取当前token"""
        if not self.current_token:
            raise ValueError("[ERROR] Token池为空")
        return self.current_token
    
    def switch_to_next(self) -> str:
        """切换到下一个token（轮询）"""
        if len(self.tokens) <= 1:
            debug_log("[WARN] 只有一个token，无法切换")
            return self.current_token
        
        # 切换到下一个token
        self.current_index = (self.current_index + 1) % len(self.tokens)
        self.current_token = self.tokens[self.current_index]
        
        debug_log(f"[SWITCH] 切换到下一个token (索引: {self.current_index}/{len(self.tokens)}): {self.current_token[:20]}...")
        return self.current_token
    
    def get_pool_size(self) -> int:
        """获取token池大小"""
        return len(self.tokens)
    
    def reload(self):
        """重新加载token池"""
        debug_log("[RELOAD] 重新加载token池")
        self._load_tokens()


# 全局token池实例（单例模式）
_token_pool_instance: Optional[TokenPool] = None


def get_token_pool() -> TokenPool:
    """获取全局token池实例"""
    global _token_pool_instance
    if _token_pool_instance is None:
        _token_pool_instance = TokenPool()
    return _token_pool_instance

