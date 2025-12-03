#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像处理模块 - 支持图像上传到Z.AI
"""

import base64
import mimetypes
import httpx
from typing import Dict, Any, Optional, Tuple
from io import BytesIO

from .helpers import error_log, info_log, debug_log


def is_base64_image(url: str) -> bool:
    """
    判断是否是base64编码的图像
    
    Args:
        url: 图像URL或base64字符串
        
    Returns:
        是否是base64图像
    """
    return url.startswith("data:image/")


def decode_base64_image(base64_str: str) -> Tuple[bytes, str, str]:
    """
    解码base64图像数据
    
    Args:
        base64_str: base64编码的图像字符串 (data:image/png;base64,...)
        
    Returns:
        (image_bytes, content_type, extension)
    """
    try:
        # 解析data URL格式: data:image/png;base64,iVBORw0KG...
        if base64_str.startswith("data:"):
            # 分离头部和数据
            header, data = base64_str.split(",", 1)
            
            # 提取content type
            content_type = header.split(":")[1].split(";")[0]
            
            # 从content type推断扩展名
            extension = mimetypes.guess_extension(content_type) or ".png"
            if extension.startswith("."):
                extension = extension[1:]
            
            # 解码base64数据
            image_bytes = base64.b64decode(data)
            
            debug_log(
                "解码base64图像",
                content_type=content_type,
                extension=extension,
                size=len(image_bytes)
            )
            
            return image_bytes, content_type, extension
        else:
            raise ValueError("不是有效的data URL格式")
    
    except Exception as e:
        debug_log(f"解码base64图像失败: {e}")
        raise


async def download_image_from_url(url: str, http_client: Optional[httpx.AsyncClient] = None) -> Tuple[bytes, str, str]:
    """
    从URL下载图像

    Args:
        url: 图像URL
        http_client: 可选的HTTP客户端(复用连接池)

    Returns:
        (image_bytes, content_type, extension)
    """
    try:
        # 使用外部传入的HTTP客户端,如果没有则创建临时客户端
        if http_client:
            client = http_client
            should_close = False
        else:
            client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
            should_close = True

        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()

            image_bytes = response.content
            content_type = response.headers.get("Content-Type", "image/png")

            # 推断扩展名
            extension = mimetypes.guess_extension(content_type) or ".png"
            if extension.startswith("."):
                extension = extension[1:]

            debug_log(
                "从URL下载图像",
                url=url[:100],
                content_type=content_type,
                extension=extension,
                size=len(image_bytes)
            )

            return image_bytes, content_type, extension

        finally:
            # 如果是临时创建的客户端,需要关闭
            if should_close:
                await client.aclose()

    except Exception as e:
        error_log(f"从URL下载图像失败: {e}", url=url[:100])
        raise


async def upload_image_to_zai(
    image_bytes: bytes,
    filename: str,
    content_type: str,
    token: str,
    client: httpx.AsyncClient
) -> Dict[str, Any]:
    """
    上传图像到Z.AI文件API
    
    Args:
        image_bytes: 图像字节数据
        filename: 文件名
        content_type: 内容类型
        token: Z.AI认证令牌
        client: httpx客户端
        
    Returns:
        上传后的文件信息
    """
    try:
        # 构建multipart/form-data请求
        files = {
            "file": (filename, BytesIO(image_bytes), content_type)
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Referer": "https://chat.z.ai/",
            "Origin": "https://chat.z.ai",
        }
        
        # 上传文件
        info_log("开始上传图像到Z.AI", filename=filename, size=len(image_bytes))
        
        response = await client.post(
            "https://chat.z.ai/api/v1/files/",
            files=files,
            headers=headers,
            timeout=60.0
        )
        
        response.raise_for_status()
        file_data = response.json()
        
        info_log(
            "图像上传成功",
            file_id=file_data.get("id"),
            filename=file_data.get("filename"),
            size=file_data.get("meta", {}).get("size")
        )
        
        return file_data
    
    except Exception as e:
        error_log(f"上传图像到Z.AI失败: {e}")
        raise


def format_file_for_zai_request(file_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    格式化文件数据为Z.AI请求格式
    
    Args:
        file_data: 上传后的文件数据
        
    Returns:
        格式化后的文件对象
    """
    import uuid
    return {
        "type": "image",
        "file": file_data,
        "id": file_data["id"],
        "url": f"/api/v1/files/{file_data['id']}/content",
        "name": file_data.get("filename", "image"),
        "status": "uploaded",
        "size": file_data.get("meta", {}).get("size", 0),
        "error": "",
        "itemId": str(uuid.uuid4()),  # 使用独立的UUID
        "media": "image"
        # ref_user_msg_id 将在 zai_transformer.py 中添加
    }


async def process_image_content(
    image_url: str,
    token: str,
    client: httpx.AsyncClient
) -> Optional[Dict[str, Any]]:
    """
    处理图像内容：从base64或URL获取图像，上传到Z.AI，返回文件对象
    
    Args:
        image_url: 图像URL或base64字符串
        token: Z.AI认证令牌
        client: httpx客户端
        
    Returns:
        格式化后的文件对象，如果处理失败则返回None
    """
    try:
        # 判断是base64还是URL
        if is_base64_image(image_url):
            debug_log("检测到base64编码图像")
            image_bytes, content_type, extension = decode_base64_image(image_url)
            filename = f"image.{extension}"
        else:
            debug_log("检测到URL图像", url=image_url[:100])
            # 复用HTTP客户端,避免重复创建连接
            image_bytes, content_type, extension = await download_image_from_url(image_url, http_client=client)
            # 从URL提取文件名
            from urllib.parse import urlparse
            parsed_url = urlparse(image_url)
            path_parts = parsed_url.path.split("/")
            filename = path_parts[-1] if path_parts[-1] else f"image.{extension}"
            # 确保文件名有正确的扩展名
            if not filename.endswith(f".{extension}"):
                filename = f"{filename}.{extension}"
        
        # 上传到Z.AI
        file_data = await upload_image_to_zai(
            image_bytes=image_bytes,
            filename=filename,
            content_type=content_type,
            token=token,
            client=client
        )
        
        # 格式化为Z.AI请求格式
        return format_file_for_zai_request(file_data)
    
    except Exception as e:
        error_log(f"处理图像内容失败: {e}", image_url=image_url[:100])
        return None

