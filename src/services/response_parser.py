#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
响应解析器模块 - 封装所有响应内容解析逻辑

从 openai_service.py 拆分出来，提供统一的响应内容处理接口

性能优化：所有正则表达式在类初始化时预编译
"""

import re
from typing import Tuple, List

from ..helpers import debug_log


class ResponseParser:
    """响应解析器类，封装所有响应内容解析逻辑"""

    def __init__(self):
        """初始化解析器，预编译所有正则表达式"""
        # ===== clean_thinking 使用的正则表达式 =====
        # 属性残片检测
        self._re_attr_residue = re.compile(r'(duration=|last_tool_call_name|view=)')
        self._re_attr_end = re.compile(r'[">]$')
        # 标签清理
        self._re_glm_block = re.compile(r'<glm_block[^>]*>.*?</glm_block>', re.DOTALL)
        self._re_url_tag = re.compile(r'<url>[^<]*</url>')
        self._re_details_open = re.compile(r'<details[^>]*>')
        self._re_details_close = re.compile(r'</details>')
        self._re_summary = re.compile(r'<summary[^>]*>.*?</summary>', re.DOTALL)
        # 引用标记清理
        self._re_quote_line_start = re.compile(r'^>\s*', re.MULTILINE)
        self._re_quote_newline = re.compile(r'\n>\s*')
        # 多余空行清理
        self._re_multi_newlines = re.compile(r'\n{3,}')
        
        # ===== extract_image_urls 使用的正则表达式 =====
        # 格式1: image_url 类型（转义格式）
        self._re_image_url_escaped = re.compile(r'\\\"url\\\":\s*\\\"(https?://[^\\\"\\\\]+(?:\\\\.[^\\\"\\\\]+)*[^\\\"\\\\]*)\\\"')
        self._re_image_url_plain = re.compile(r'"url":\s*"(https?://[^"]+)"')
        # 格式2: img_url 类型（转义格式）
        self._re_img_url_escaped = re.compile(r'\\\"img_url\\\":\s*\\\"(https?://[^\\\"]+)\\\"')
        self._re_img_url_plain = re.compile(r'"img_url":\s*"(https?://[^"]+)"')
        
        # ===== extract_search_info 使用的正则表达式 =====
        self._re_queries = re.compile(r'"queries":\s*\[(.*?)\]')
        self._re_query_items = re.compile(r'"([^"]+)"')

    def clean_thinking(self, delta_content: str) -> str:
        """清理 thinking 内容，提取纯文本
        
        处理格式：
        - 移除 <details> 和 <summary> 标签
        - 移除 markdown 引用符号 "> "
        - 保留纯文本内容
        """
        if not delta_content:
            return ""
        
        # 0. 先丢弃可能出现在 <details> 之前的属性残片
        first_newline = delta_content.find("\n")
        if first_newline != -1:
            first_line = delta_content[:first_newline].strip()
            if self._re_attr_residue.search(first_line) and self._re_attr_end.search(first_line):
                delta_content = delta_content[first_newline + 1:]

        # 1. 移除 <glm_block>...</glm_block> 工具调用块
        delta_content = self._re_glm_block.sub('', delta_content)
        
        # 2. 移除 <url>...</url> 标签
        delta_content = self._re_url_tag.sub('', delta_content)

        # 3. 移除 <details> 开始标签
        delta_content = self._re_details_open.sub('', delta_content)
        
        # 4. 移除 </details> 结束标签
        delta_content = self._re_details_close.sub('', delta_content)

        # 5. 移除 <summary> 标签及其内容
        delta_content = self._re_summary.sub('', delta_content)
        
        # 6. 移除行首的引用标记 "> "
        delta_content = self._re_quote_line_start.sub('', delta_content)
        delta_content = self._re_quote_newline.sub('\n', delta_content)
        
        # 7. 移除多余的空行
        delta_content = self._re_multi_newlines.sub('\n\n', delta_content)
        
        # 8. 去除首尾空白
        return delta_content.strip()

    def split_edit_content(self, edit_content: str) -> Tuple[str, str]:
        """拆分 edit_content，返回 (thinking_part, answer_part)
        
        处理格式：
        <details type="reasoning" done="false/true" ...>
        <summary>Thinking...</summary>
        > 思考内容
        </details>
        回答内容
        """
        if not edit_content:
            return "", ""

        thinking_part = ""
        answer_part = ""

        if "</details>" in edit_content:
            parts = edit_content.split("</details>", 1)
            thinking_part = parts[0] + "</details>"
            answer_part = parts[1] if len(parts) > 1 else ""
        else:
            answer_part = edit_content

        # 清理 thinking 内容
        if thinking_part:
            thinking_part = self.clean_thinking(thinking_part)
        
        # 清理 answer 内容
        answer_part = answer_part.strip()
        if answer_part:
            answer_part = answer_part.lstrip('\n')
            answer_part = answer_part.replace("<think>", "").replace("</think>", "")
        
        return thinking_part, answer_part

    def diff_new_content(self, existing: str, incoming: str) -> str:
        """计算 incoming 相比 existing 的新增部分（用于流式增量输出）"""
        incoming = incoming or ""
        if not incoming:
            return ""

        existing = existing or ""
        if not existing:
            return incoming

        if incoming == existing:
            return ""

        # 如果 incoming 是 existing 的扩展，返回新增部分
        if incoming.startswith(existing):
            return incoming[len(existing):]

        # 寻找最长公共前缀以计算增量
        max_overlap = min(len(existing), len(incoming))
        for overlap in range(max_overlap, 0, -1):
            if existing[-overlap:] == incoming[:overlap]:
                return incoming[overlap:]

        # 如果 existing 完全包含在 incoming 中
        if existing in incoming:
            return incoming.replace(existing, "", 1)

        # 无法确定增量，返回完整内容
        return incoming

    def extract_image_urls(self, content: str) -> List[str]:
        """从上游响应内容中提取图片URL
        
        处理格式示例：
        1. {"image_url":{"url":"https://qc4n.bigmodel.cn/xxx.png?..."}}
        2. {"img_url": "https://bigmodel-us3-prod-agent.cn-wlcb.ufileos.com/xxx.jpg", ...}
        
        Returns:
            list: 提取到的图片URL列表
        """
        if not content:
            return []
        
        image_urls = []
        
        # === 格式1: image_url 类型（bigmodel.cn 域名）===
        if '\\"type\\":\\"image_url\\"' in content or '"type":"image_url"' in content:
            matches = self._re_image_url_escaped.findall(content)
            for url in matches:
                clean_url = url.replace('\\/', '/').replace('\\"', '"')
                if clean_url and 'bigmodel.cn' in clean_url:
                    image_urls.append(clean_url)
            
            if not image_urls:
                matches = self._re_image_url_plain.findall(content)
                for url in matches:
                    if url and 'bigmodel.cn' in url:
                        image_urls.append(url)
        
        # === 格式2: img_url 类型（ufileos.com 域名）===
        if 'img_url' in content or 'image_reference' in content:
            matches = self._re_img_url_escaped.findall(content)
            for url in matches:
                clean_url = url.replace('\\/', '/')
                if clean_url and ('ufileos.com' in clean_url or 'bigmodel' in clean_url):
                    image_urls.append(clean_url)
            
            if not image_urls:
                matches = self._re_img_url_plain.findall(content)
                for url in matches:
                    if url and ('ufileos.com' in url or 'bigmodel' in url):
                        image_urls.append(url)
        
        return image_urls

    def format_images_as_markdown(self, image_urls: List[str]) -> str:
        """将图片URL列表格式化为markdown图片格式
        
        Args:
            image_urls: 图片URL列表
            
        Returns:
            str: markdown格式的图片字符串
        """
        if not image_urls:
            return ""
        
        markdown_images = []
        for i, url in enumerate(image_urls, 1):
            markdown_images.append(f"![图片{i}]({url})")
        
        return "\n\n".join(markdown_images)

    def extract_search_info(self, reasoning_content: str, edit_content: str) -> str:
        """从 edit_content 中提取搜索信息"""
        if edit_content and "<glm_block" in edit_content and "search" in edit_content:
            try:
                decoded = edit_content
                try:
                    decoded = edit_content.encode("utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")
                except Exception:
                    try:
                        import codecs
                        decoded = codecs.decode(edit_content, "unicode_escape")
                    except Exception:
                        pass

                queries_match = self._re_queries.search(decoded)
                if queries_match:
                    queries_str = queries_match.group(1)
                    queries = self._re_query_items.findall(queries_str)
                    if queries:
                        search_info = "🔍 **搜索：** " + "　".join(queries[:5])
                        reasoning_content += f"\n\n{search_info}\n\n"
                        debug_log("[搜索信息] 提取到搜索查询", queries=queries)
            except Exception as exc:
                debug_log("[搜索信息] 提取失败", error=str(exc))
        return reasoning_content


# 全局单例实例
response_parser = ResponseParser()
