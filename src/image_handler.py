#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像处理模块 - 支持图像上传到Z.AI
"""

import base64
import hashlib
import mimetypes
import uuid
from io import BytesIO
from typing import Dict, Any, Optional, Tuple

import httpx

from .helpers import error_log, info_log, debug_log


MINIMAL_FILE_META_KEYS = (
    "id",
    "filename",
    "content_type",
    "mimetype",
    "extension",
    "meta",
)


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


async def fetch_image_bytes_and_meta(
    image_url: str,
    client: Optional[httpx.AsyncClient] = None,
) -> Tuple[bytes, str, str, str]:
    """统一获取图片字节、内容类型、扩展名和文件名。"""
    if is_base64_image(image_url):
        debug_log("检测到base64编码图像")
        image_bytes, content_type, extension = decode_base64_image(image_url)
        filename = f"image.{extension}"
        return image_bytes, content_type, extension, filename

    debug_log("检测到URL图像", url=image_url[:100])
    image_bytes, content_type, extension = await download_image_from_url(image_url, http_client=client)
    from urllib.parse import urlparse

    parsed_url = urlparse(image_url)
    path_parts = parsed_url.path.split("/")
    filename = path_parts[-1] if path_parts[-1] else f"image.{extension}"
    if not filename.endswith(f".{extension}"):
        filename = f"{filename}.{extension}"
    return image_bytes, content_type, extension, filename



def compute_image_asset_key(image_bytes: bytes) -> str:
    """基于图片实际二进制计算稳定哈希，用于同会话内复用。"""
    return hashlib.sha256(image_bytes).hexdigest()


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



def build_minimal_file_data(
    file_id: str,
    name: str = "image",
    size: int = 0,
    content_type: Optional[str] = None,
) -> Dict[str, Any]:
    """根据轻量持久化元数据重建最小可用 file_data。"""
    normalized_content_type = content_type or "image/png"
    return {
        "id": file_id,
        "filename": name or "image",
        "content_type": normalized_content_type,
        "mimetype": normalized_content_type,
        "extension": ((name.rsplit(".", 1)[-1] if "." in name else "png") or "png"),
        "meta": {
            "size": int(size or 0),
        },
    }



def format_file_for_zai_request(file_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    格式化文件数据为Z.AI请求格式
    
    Args:
        file_data: 上传后的文件数据
        
    Returns:
        格式化后的文件对象
    """
    return {
        "type": "image",
        "file": file_data,
        "id": file_data["id"],
        "url": f"/api/v1/files/{file_data['id']}",
        "name": file_data.get("filename", "image"),
        "status": "uploaded",
        "size": file_data.get("meta", {}).get("size", 0),
        "error": "",
        "itemId": str(uuid.uuid4()),
        "media": "image"
        # ref_user_msg_id 将在 zai_transformer.py 中添加
    }



def minimal_file_data_from_uploaded(file_data: Dict[str, Any]) -> Dict[str, Any]:
    """裁剪上传返回的 file_data，仅保留后续复用所需最小字段。"""
    minimal: Dict[str, Any] = {}
    for key in MINIMAL_FILE_META_KEYS:
        if key in file_data:
            minimal[key] = file_data[key]
    if "meta" not in minimal or not isinstance(minimal.get("meta"), dict):
        minimal["meta"] = {"size": 0}
    return minimal



def build_reusable_file_payload(asset: Dict[str, Any], ref_user_msg_id: Optional[str] = None) -> Dict[str, Any]:
    """根据会话缓存中的图片资产重建当前请求所需的 file payload。"""
    raw_file_data = asset.get("file_data") if isinstance(asset.get("file_data"), dict) else None
    if raw_file_data:
        file_data = raw_file_data
    else:
        file_data = build_minimal_file_data(
            file_id=str(asset.get("file_id") or ""),
            name=str(asset.get("name") or "image"),
            size=int(asset.get("size") or 0),
        )

    payload = format_file_for_zai_request(file_data)
    if ref_user_msg_id:
        payload["ref_user_msg_id"] = ref_user_msg_id
    return payload


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
        image_bytes, content_type, extension, filename = await fetch_image_bytes_and_meta(image_url, client)
        
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
