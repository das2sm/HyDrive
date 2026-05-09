import xml.etree.ElementTree as ET
import json

# 读取源 XML
src_tree = ET.parse('leaderboard/data/bench2drive220.xml')
src_root = src_tree.getroot()

# 读取 Ablation.json
with open('leaderboard/data/test.json', 'r', encoding='utf-8') as f:
    ablation_ids = set(sum(json.load(f).values(), []))   # 合并所有难度分组

# 创建新的根节点
new_root = ET.Element('routes')

# 遍历原 XML，保留指定 id 的 route
for route in src_root.findall('route'):
    route_id = int(route.get('id'))
    if route_id in ablation_ids:
        new_root.append(route)          # 直接追加节点

# 写出到新文件
ET.ElementTree(new_root).write('leaderboard/data/test.xml', encoding='utf-8')