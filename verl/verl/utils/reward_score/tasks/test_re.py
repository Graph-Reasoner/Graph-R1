import re

def extract_nodes(text):
    # 定义正则表达式模式，允许名字中包含空格
    # pattern = r"Please identify the common neighbors of\s+(.*?)\s+and\s+(.*?)\s+in this network\."
    pattern = r"Please determine the shortest path between\s+(.*?)\s+and\s+(.*?)\s+in this network."
    # 使用正则表达式匹配
    match = re.search(pattern, text)
    
    # 如果匹配成功，返回两个节点
    if match:
        node1 = match.group(1)
        node2 = match.group(2)
        return node1, node2
    else:
        return None, None

# 测试代码
# test_text = "Please identify the common neighbors of Thorsten Wild and Gerald Matz in this network."
test_text = "Please determine the shortest path between Texas and United States in this network."
node1, node2 = extract_nodes(test_text)
# print(f"Node 1: {node1}")
# print(f"Node 2: {node2}")