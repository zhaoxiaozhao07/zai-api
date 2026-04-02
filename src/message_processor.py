#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
消息处理器模块 - 封装 OpenAI 消息格式处理逻辑

从 zai_transformer.py 拆分出来，提供统一的消息处理接口
"""

from typing import Dict, List, Tuple, Any

from .helpers import debug_log


class MessageProcessor:
    """
    消息处理器类
    
    封装所有消息处理逻辑，包括：
    - system 角色转换
    - 图像内容提取
    - 用户消息提取
    """
    
    def has_image_content(self, messages: List[Dict]) -> bool:
        """
        检测消息中是否包含图像
        
        Args:
            messages: 消息列表
            
        Returns:
            bool: 如果消息中包含图像内容则返回 True
        """
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    for part in content:
                        if (part.get("type") == "image_url" and
                            part.get("image_url", {}).get("url")):
                            return True
        return False
    
    def process_messages(self, messages: list, is_vision_model: bool = False) -> Tuple[list, list]:
        """
        处理消息列表，转换 system 角色和提取图片内容
        
        Args:
            messages: 原始消息列表
            is_vision_model: 是否是视觉模型（GLM-4.6V），视觉模型保留图片在 messages 中
            
        Returns:
            (处理后的消息列表, 图片 URL 列表)
        """
        processed_messages = []
        image_urls = []
        
        for idx, orig_msg in enumerate(messages):
            msg = orig_msg.copy()

            # 处理 system 角色转换
            if msg.get("role") == "system":
                msg["role"] = "user"
                content = msg.get("content")

                if isinstance(content, list):
                    msg["content"] = [
                        {"type": "text", "text": "This is a system command, you must enforce compliance."}
                    ] + content
                elif isinstance(content, str):
                    msg["content"] = f"This is a system command, you must enforce compliance.{content}"

            # 处理 user 角色的图片内容
            elif msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    new_content = []
                    for part_idx, part in enumerate(content):
                        if (
                            part.get("type") == "image_url"
                            and part.get("image_url", {}).get("url")
                            and isinstance(part["image_url"]["url"], str)
                        ):
                            image_url = part["image_url"]["url"]
                            debug_log(f"    消息[{idx}]内容[{part_idx}]: 检测到图片URL")
                            image_urls.append(image_url)
                            
                            # 视觉模型：保留图片在消息中
                            if is_vision_model:
                                new_content.append(part)
                            # 非视觉模型：移除图片内容，只保留文本
                        elif part.get("type") == "text":
                            new_content.append(part)
                    
                    # 如果 new_content 只有文本，提取为字符串
                    if len(new_content) == 1 and new_content[0].get("type") == "text":
                        msg["content"] = new_content[0].get("text", "")
                    elif new_content:
                        msg["content"] = new_content
                    else:
                        msg["content"] = ""

            processed_messages.append(msg)
        
        return processed_messages, image_urls
    
    def replace_image_urls_with_file_ids(
        self,
        messages: List[Dict],
        image_asset_keys: List[str],
        assets_by_key: Dict[str, Dict[str, Any]],
        latest_message_only: bool = True,
    ) -> List[Dict]:
        """将用户消息中的图片 base64/URL 替换为已上传文件 file_id，避免把大图数据继续带入 completions body。"""
        if not messages or not image_asset_keys:
            return messages

        replaced_messages = [msg.copy() if isinstance(msg, dict) else msg for msg in messages]

        target_message_indexes = [len(replaced_messages) - 1] if latest_message_only else list(range(len(replaced_messages)))
        image_file_ids: List[str] = []
        for asset_key in image_asset_keys:
            asset = assets_by_key.get(asset_key)
            file_id = asset.get("file_id") if isinstance(asset, dict) else None
            if isinstance(file_id, str) and file_id.strip():
                image_file_ids.append(file_id.strip())

        if not image_file_ids:
            return messages

        image_index = 0
        for message_index in target_message_indexes:
            latest_message = replaced_messages[message_index]
            if not isinstance(latest_message, dict) or latest_message.get("role") != "user":
                continue

            content = latest_message.get("content")
            if not isinstance(content, list):
                continue

            rebuilt_content: List[Dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    rebuilt_content.append(part)
                    continue

                if part.get("type") == "image_url":
                    if image_index >= len(image_file_ids):
                        continue
                    rebuilt_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_file_ids[image_index],
                            },
                        }
                    )
                    image_index += 1
                    continue

                rebuilt_content.append(part)

            latest_message["content"] = rebuilt_content
            replaced_messages[message_index] = latest_message

            if image_index >= len(image_file_ids):
                break

        return replaced_messages

    def extract_last_user_content(self, messages: list) -> str:
        """
        提取最后一条用户消息的文本内容
        
        Args:
            messages: 消息列表
            
        Returns:
            最后一条用户消息的文本内容
        """
        user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_content = content
                elif isinstance(content, list) and len(content) > 0:
                    for part in content:
                        if part.get("type") == "text":
                            user_content = part.get("text", "")
                            break
                break
        return user_content


# 全局单例实例
message_processor = MessageProcessor()
