# -*- coding: utf-8 -*-
"""
Phase 1 重构脚本：删除第 3 章 + 章节号/表号/公式号/交叉引用批量重编号

映射规则:
  - 删除整个原 # 3 章
  - 原 4.X → 新 3.Y（非线性，因 LLVIP 将作为新 3.3 插入）：
    4.1 → 3.1
    4.2 → 3.2 (RoadScene 保留位置)
    -- 新 3.3 = LLVIP（StrReplace 阶段插入，本脚本不处理）
    4.3 → 3.4
    4.4 → 3.5
    4.5 → 3.6
    4.6 → 3.7
    4.7 → 3.8
    4.8 → 3.9
  - 原 5.X → 新 4.X（一对一）
  - 原 6.X → 新 5.X（一对一）
  - 表 4-X → 表 3-X
  - 表 5-X → 表 4-X
  - 公式 (4-X) → (3-X)
  - 公式 \tag{4-X} → \tag{3-X}
"""
import re
import sys

PATH = r'c:\Users\abirdlikefish\Desktop\毕设\efficient_loftr\abfNote\毕业论文.md'

with open(PATH, encoding='utf-8') as f:
    content = f.read()

orig_len = len(content)

# ---- Step 1: 删除原第 3 章 (从 "# 3 基于..." 到 "# 4 数据集与实验通用设计" 之前) ----
# 用非贪婪 + lookahead 锚点
pattern_ch3 = r'# 3 基于 Efficient LoFTR 的跨模态匹配方法.*?(?=# 4 数据集与实验通用设计)'
m = re.search(pattern_ch3, content, flags=re.DOTALL)
if m:
    print(f'Step 1: 删除第 3 章 ({m.end()-m.start()} chars)')
    content = re.sub(pattern_ch3, '', content, flags=re.DOTALL)
else:
    print('WARN: 未找到第 3 章模式!')
    sys.exit(1)

# ---- Step 2: 第 4 章子节 4.X → 新 3.Y (非线性映射，先大后小避免冲突) ----
ch4_subnode_map = [(8, 9), (7, 8), (6, 7), (5, 6), (4, 5), (3, 4), (2, 2), (1, 1)]

count_step2 = 0
for old_x, new_x in ch4_subnode_map:
    # 三级标题: ### 4.X.Y -> ### 3.NEW_X.Y
    pat3 = rf'^### 4\.{old_x}\.'
    n = len(re.findall(pat3, content, flags=re.MULTILINE))
    content = re.sub(pat3, f'### 3.{new_x}.', content, flags=re.MULTILINE)
    count_step2 += n

    # 二级标题: ## 4.X (后面不能再跟数字) -> ## 3.NEW_X
    pat2 = rf'^## 4\.{old_x}(?![\d])'
    n = len(re.findall(pat2, content, flags=re.MULTILINE))
    content = re.sub(pat2, f'## 3.{new_x}', content, flags=re.MULTILINE)
    count_step2 += n

# ---- Step 3: 第 4 章三级正文交叉引用 "4.X.Y 节/小节/末/中" 与 二级 "4.X 节/中/介绍/见" 等 ----
# 三级先处理（更长 pattern 优先）
count_step3 = 0
for old_x, new_x in ch4_subnode_map:
    # 4.X.Y -> 3.NEW_X.Y (正文中)
    # 用 lookbehind 确保前面不是数字（避免 "8.4.7.X" 这种），lookahead 确保后面跟中文标点或空格
    pat = rf'(?<![\d.])4\.{old_x}\.(\d+)(?![\d.])'
    matches = re.findall(pat, content)
    n = len(matches)
    content = re.sub(pat, lambda m, nx=new_x: f'3.{nx}.{m.group(1)}', content)
    count_step3 += n

    # 4.X (二级) -> 3.NEW_X
    pat = rf'(?<![\d.])4\.{old_x}(?![\d.])'
    n = len(re.findall(pat, content))
    content = re.sub(pat, f'3.{new_x}', content)
    count_step3 += n

# ---- Step 4: 第 4 章顶级标题 "# 4 " -> "# 3 " ----
n = len(re.findall(r'^# 4 ', content, flags=re.MULTILINE))
content = re.sub(r'^# 4 ', '# 3 ', content, flags=re.MULTILINE)
print(f'Step 4: # 4 → # 3 ({n} matches)')

# "第 4 章" -> "第 3 章"
n = content.count('第 4 章')
content = content.replace('第 4 章', '第 3 章')
print(f'Step 4b: 第 4 章 → 第 3 章 ({n} matches)')

# ---- Step 5: 第 5 章子节 5.X → 4.X (一对一) ----
n = len(re.findall(r'^### 5\.', content, flags=re.MULTILINE))
content = re.sub(r'^### 5\.', '### 4.', content, flags=re.MULTILINE)
n2 = len(re.findall(r'^## 5\.', content, flags=re.MULTILINE))
content = re.sub(r'^## 5\.', '## 4.', content, flags=re.MULTILINE)
n3 = len(re.findall(r'^# 5 ', content, flags=re.MULTILINE))
content = re.sub(r'^# 5 ', '# 4 ', content, flags=re.MULTILINE)
print(f'Step 5: 5.X 标题 → 4.X (### {n}, ## {n2}, # {n3})')

# 正文 5.X.Y / 5.X 引用 -> 4.X.Y / 4.X
pat_5xy = r'(?<![\d.])5\.(\d+)\.(\d+)(?![\d.])'
n = len(re.findall(pat_5xy, content))
content = re.sub(pat_5xy, r'4.\1.\2', content)
pat_5x = r'(?<![\d.])5\.(\d+)(?![\d.])'
n2 = len(re.findall(pat_5x, content))
content = re.sub(pat_5x, r'4.\1', content)
print(f'Step 5b: 5.X 引用 → 4.X (5.X.Y {n}, 5.X {n2})')

# 第 5 章 -> 第 4 章
n = content.count('第 5 章')
content = content.replace('第 5 章', '第 4 章')
print(f'Step 5c: 第 5 章 → 第 4 章 ({n} matches)')

# ---- Step 6: 第 6 章子节 6.X → 5.X (一对一) ----
n = len(re.findall(r'^### 6\.', content, flags=re.MULTILINE))
content = re.sub(r'^### 6\.', '### 5.', content, flags=re.MULTILINE)
n2 = len(re.findall(r'^## 6\.', content, flags=re.MULTILINE))
content = re.sub(r'^## 6\.', '## 5.', content, flags=re.MULTILINE)
n3 = len(re.findall(r'^# 6 ', content, flags=re.MULTILINE))
content = re.sub(r'^# 6 ', '# 5 ', content, flags=re.MULTILINE)
print(f'Step 6: 6.X 标题 → 5.X (### {n}, ## {n2}, # {n3})')

pat_6xy = r'(?<![\d.])6\.(\d+)\.(\d+)(?![\d.])'
n = len(re.findall(pat_6xy, content))
content = re.sub(pat_6xy, r'5.\1.\2', content)
pat_6x = r'(?<![\d.])6\.(\d+)(?![\d.])'
n2 = len(re.findall(pat_6x, content))
content = re.sub(pat_6x, r'5.\1', content)
print(f'Step 6b: 6.X 引用 → 5.X (6.X.Y {n}, 6.X {n2})')

n = content.count('第 6 章')
content = content.replace('第 6 章', '第 5 章')
print(f'Step 6c: 第 6 章 → 第 5 章 ({n} matches)')

# ---- Step 7: 表号 ----
n = len(re.findall(r'表 4-\d+', content))
content = re.sub(r'表 4-(\d+)', r'表 3-\1', content)
print(f'Step 7: 表 4-X → 表 3-X ({n} matches)')

n = len(re.findall(r'表 5-\d+', content))
content = re.sub(r'表 5-(\d+)', r'表 4-\1', content)
print(f'Step 7b: 表 5-X → 表 4-X ({n} matches)')

# ---- Step 8: 公式号 (4-X) → (3-X) 与 \tag{4-X} → \tag{3-X} ----
n = len(re.findall(r'\(4-\d+\)', content))
content = re.sub(r'\(4-(\d+)\)', r'(3-\1)', content)
print(f'Step 8: (4-X) → (3-X) ({n} matches)')

n = len(re.findall(r'\\tag\{4-\d+\}', content))
content = re.sub(r'\\tag\{4-(\d+)\}', r'\\tag{3-\1}', content)
print(f'Step 8b: \\tag{{4-X}} → \\tag{{3-X}} ({n} matches)')

# 备份后写回
new_len = len(content)
print(f'\n原文件 {orig_len} chars → 新文件 {new_len} chars (Δ={new_len-orig_len})')

with open(PATH, 'w', encoding='utf-8') as f:
    f.write(content)
print('Wrote back to', PATH)
