import re


def custom_markdown_splitter(
    text: str,
    chunk_size: int = 500,
    overlap_size: int = 100,
    hash_threshold: int = 50
) -> list:
    """
    改进的 Markdown 切分函数 - 按标题切分，表格单独提取

    Args:
        text: 要切分的文本
        chunk_size: 每个 chunk 的目标大小
        overlap_size: 重叠大小
        hash_threshold: 标题后不切分的距离（暂不使用，保留参数兼容性）

    Returns:
        切分后的文本列表
    """
    chunks = []

    # 1. 先提取所有表格，用占位符替换
    table_pattern = re.compile(r'<html><body><table>.*?</table></body></html>', re.DOTALL)
    tables = []
    table_placeholders = []

    def save_table(match):
        table_content = match.group(0)
        tables.append(table_content)
        placeholder = f"__TABLE_PLACEHOLDER_{len(tables) - 1}__"
        table_placeholders.append(placeholder)
        return placeholder

    text_without_tables = table_pattern.sub(save_table, text)

    # 2. 按标题切分文本（支持一级、二级、三级标题）
    # 匹配 # ## ### 开头的行
    header_pattern = re.compile(r'^#{1,3}\s+.+$', re.MULTILINE)

    # 找到所有标题位置
    headers = []
    for match in header_pattern.finditer(text_without_tables):
        header_text = match.group(0).strip()
        header_start = match.start()
        headers.append((header_start, header_text))

    if headers:
        first_header_start = headers[0][0]
        before_first_header = text_without_tables[:first_header_start].strip()
        if before_first_header:
            # 将第一个标题之前的内容作为第一个chunk
            chunks.append(before_first_header)
            
    # 3. 按标题分段处理
    for i in range(len(headers)):
        header_start, header_text = headers[i]

        # 确定当前段的结束位置
        if i < len(headers) - 1:
            section_end = headers[i + 1][0]
        else:
            section_end = len(text_without_tables)

        # 跳过标题行本身，从内容开始
        content_start = header_start + len(header_text)

        # 提取当前标题下的内容
        section_content = text_without_tables[content_start:section_end].strip()

        if not section_content:
            continue

        # 4. 如果内容小于chunk_size，作为一个chunk
        if len(section_content) <= chunk_size:
            chunk = f"{header_text}\n\n{section_content}"
            chunks.append(chunk)
        else:
            # 5. 如果超出chunk_size，进行切分
            # 每个切分都带上标题
            sub_chunks = _split_large_section(section_content, chunk_size, header_text)
            chunks.extend(sub_chunks)

    # 6. 将表格作为单独的chunk添加
    for table in tables:
        chunks.append(table)

    return chunks


def _split_large_section(content: str, chunk_size: int, header_text: str) -> list:
    """
    切分过大的section，保持每个chunk都有标题
    """
    chunks = []
    start = 0
    content_length = len(content)

    # 定义标点符号集合
    sentence_endings = '。！？.!？'

    while start < content_length:
        # 剩余内容不足chunk_size
        if start + chunk_size >= content_length:
            remaining = content[start:].strip()
            if remaining:
                chunks.append(f"{header_text}\n\n{remaining}")
            break

        # 从 start + chunk_size 往前找最近的句末标点
        end = start + chunk_size
        curr = end - 1

        # 往前找标点，最多退 chunk_size/3 的距离
        max_backtrack = chunk_size // 3
        while curr > start and curr > end - max_backtrack:
            if content[curr] in sentence_endings:
                break
            curr -= 1

        # 如果没找到句末标点，尝试找逗号等其他分隔符
        if curr <= start or content[curr] not in sentence_endings:
            curr = end - 1
            separators = '，,、\n '
            while curr > start:
                if content[curr] in separators:
                    break
                curr -= 1

            # 如果还是没找到，就在 chunk_size 处强制切分
            if curr <= start:
                curr = end

        # 提取chunk
        chunk_content = content[start:curr + 1].strip()
        chunks.append(f"{header_text}\n\n{chunk_content}")

        # 计算下一个起点
        start = curr + 1

    return chunks




# 测试代码
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
#     plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
#     # 读取文档测试
#     with open(r"C:\Users\Arvin\Desktop\财报.md", "r", encoding="utf-8") as f:
#         text = f.read()

#     chunks = custom_markdown_splitter(text, chunk_size=800)

#     print(f"总共切分出 {len(chunks)} 个 chunks\n")

#     # 1. 计算 chunk 长度
#     chunk_lengths = [len(chunk) for chunk in chunks]
    
#     # 2. 统计信息
#     print("=== 统计信息 ===")
#     print(f"平均长度: {sum(chunk_lengths) / len(chunk_lengths):.2f}")
#     print(f"最大长度: {max(chunk_lengths)}")
#     print(f"最小长度: {min(chunk_lengths)}")
#     print(f"中位数长度: {sorted(chunk_lengths)[len(chunk_lengths) // 2]}")
    
#     # 3. 找出前5%和后5%的chunk
#     sorted_indices = sorted(range(len(chunk_lengths)), key=lambda i: chunk_lengths[i])
#     top_5_count = max(1, len(chunks) // 20)  # 至少1个
#     bottom_5_count = max(1, len(chunks) // 20)
    
#     top_5_indices = sorted_indices[-top_5_count:]  # 最长的5%
#     bottom_5_indices = sorted_indices[:bottom_5_count]  # 最短的5%
    
#     print(f"\n=== 前5%最长 chunk ({top_5_count}个) ===")
#     for i in top_5_indices:
#         print(f"\n--- Chunk {i+1} (长度: {chunk_lengths[i]}) ---")
#         print(chunks[i][:500] + "..." if len(chunks[i]) > 500 else chunks[i])
    
#     print(f"\n\n=== 后5%最短 chunk ({bottom_5_count}个) ===")
#     for i in bottom_5_indices:
#         print(f"\n--- Chunk {i+1} (长度: {chunk_lengths[i]}) ---")
#         print(chunks[i][:500] + "..." if len(chunks[i]) > 500 else chunks[i])

#     # 4. 绘制折线图
#     plt.figure(figsize=(15, 6))
#     plt.plot(range(1, len(chunk_lengths) + 1), chunk_lengths, marker='o', linestyle='-', linewidth=1, markersize=4)
#     plt.axhline(y=sum(chunk_lengths) / len(chunk_lengths), color='r', linestyle='--', label=f'平均值 ({sum(chunk_lengths) / len(chunk_lengths):.0f})')
#     plt.axhline(y=800, color='g', linestyle='--', label='目标长度 (800)')
    
#     plt.xlabel('Chunk 序号')
#     plt.ylabel('字符长度')
#     plt.title('Chunk 长度分布折线图')
#     plt.grid(True, alpha=0.3)
#     plt.legend()
    
#     # 标记前5%和后5%
#     for i in top_5_indices:
#         plt.scatter(i + 1, chunk_lengths[i], color='red', s=100, zorder=5, label='前5%' if i == top_5_indices[0] else "")
#     for i in bottom_5_indices:
#         plt.scatter(i + 1, chunk_lengths[i], color='blue', s=100, zorder=5, label='后5%' if i == bottom_5_indices[0] else "")
    
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
    
#     # 5. 额外：绘制长度分布直方图
#     plt.figure(figsize=(10, 6))
#     plt.hist(chunk_lengths, bins=20, edgecolor='black', alpha=0.7)
#     plt.axvline(x=sum(chunk_lengths) / len(chunk_lengths), color='r', linestyle='--', label=f'平均值')
#     plt.axvline(x=800, color='g', linestyle='--', label='目标长度')
#     plt.xlabel('Chunk 长度')
#     plt.ylabel('数量')
#     plt.title('Chunk 长度分布直方图')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()

