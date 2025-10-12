import httpx

def get_ip(client):
    try:
        r = client.get("https://api.ipify.org", timeout=10)
        return r.text.strip()
    except Exception as e:
        return f"请求失败：{e}"

# ===== 直连请求 =====
with httpx.Client(timeout=10) as direct_client:
    direct_ip = get_ip(direct_client)
    print(f"直连出口 IP：{direct_ip}")

# ===== 代理请求 =====
proxy = "socks5h://192.168.10.100:1083"

with httpx.Client(proxy=proxy, timeout=10) as proxy_client:
    proxy_ip = get_ip(proxy_client)
    print(f"代理出口 IP：{proxy_ip}")

# ===== 结果对比 =====
if direct_ip == proxy_ip:
    print("❌ 未走代理（出口 IP 相同）")
else:
    print("✅ 已通过代理出网")
