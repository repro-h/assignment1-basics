import os
from collections import defaultdict, Counter
import regex as re  # type: ignore
import json

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练字节级 BPE (Byte-Pair Encoding) 分词器。
    """
    
    # --- 1. 初始化基础词表 ---
    # [TODO 1]: 初始化词表 vocab，包含 0-255 的基础字节。映射关系为 ID -> bytes
    vocab = {i: bytes([i]) for i in range(256)} # 填入你的推导式
    
    num_merges = vocab_size - 256 - len(special_tokens)
    
    # --- 2. 读取语料，并按特殊 Token 分割 ---
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        special_regex = "|".join(re.escape(t) for t in special_tokens)
        parts = re.split(f"({special_regex})", text)
        # [TODO 2]: 使用列表推导式过滤 parts，去掉 special_tokens，保留纯净文本片段
        train_segments = [p for p in parts if p not in special_tokens] 
    else:
        train_segments = [text]

    # --- 3. 预分词（Pre-tokenization）并统计词频 ---
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    raw_counts = Counter()
    for segment in train_segments:
        words = gpt2_pat.findall(segment)
        for word in words:
            # [TODO 3]: 将 word 转换为 UTF-8 字节流，并将单字节对象打包成不可变的 tuple，存入 raw_counts
            raw_counts[tuple(bytes([b]) for b in word.encode('UTF-8'))] += 1
            
            
    # --- 构建高效数据结构以支持快速合并 ---
    words_list = []
    counts_list = []
    # [TODO 4]: 遍历 raw_counts，将不可变的 tuple 转换为可修改的 list 存入 words_list，频率存入 counts_list
    for word_tuple, freq in raw_counts.items():
        words_list.append(list(word_tuple))
        counts_list.append(freq)
    
    stats = defaultdict(int)
    indices = defaultdict(set)
    
    # [TODO 5]: 局部提取，全局汇总！
    # 遍历 words_list 和 counts_list，初始化全局账本 stats 和倒排索引 indices
    # 提示：双重 for 循环，只在单词内部提取相邻 pair
    for idx, word in enumerate(words_list):
        freq = counts_list[idx]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            stats[pair] += freq
            indices[pair].add(idx)
    
    merges = [] 

    # --- 4. 迭代合并流程 (核心战场) ---
    for _ in range(num_merges):
        if not stats:
            break
            
        # [TODO 6]: 寻找全场最高频 pair (注意平局时按字典序比较)
        best_pair = max(stats.items(), key=lambda x : (x[1], x[0]))[0]
        
        # [TODO 7]: 边界防御：如果 best_pair 的频率 <= 0，应该做什么？
        if stats[best_pair] <= 0:
            break
        
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        
        # [TODO 8]: 特种部队拿名单！从倒排索引获取需要空降的单词下标集合
        # 极度危险提示：记得拍快照！
        relevant_indices = list(indices[best_pair]) 
        
        # --- 4c. 遍历并执行精确爆破 ---
        for idx in relevant_indices:
            word = words_list[idx]
            freq = counts_list[idx]
            
            i = 0
            while i < len(word) - 1:
                # [TODO 9]: 敲门检查！验证当前位置真的匹配 best_pair
                if word[i] == best_pair[0] and word[i + 1] == best_pair[1]: # 替换为真实的检查逻辑
                    
                    # 动作 A: 破坏旧账本 (左邻居和右邻居的频率扣除)
                    # 提示：注意边界，减到 0 时记得 del
                    if i > 0:
                        prev_pair = (word[i - 1], word[i])
                        stats[prev_pair] -= freq
                        if stats[prev_pair] == 0:
                            del stats[prev_pair]
                    
                    if i < len(word) - 2:
                        prev_pair = (word[i + 1], word[i + 2])
                        stats[prev_pair] -= freq
                        if stats[prev_pair] == 0:
                            del stats[prev_pair]
                    
                    # 动作 B: 物理切除与合并
                    # 提示：原地修改 word 列表
                    word[i] = new_token
                    del word[i + 1]
                    
                    # 动作 C: 登记新账本和新索引 (新产生的左邻居和右邻居)
                    # 提示：频率增加，倒排索引 add(idx)
                    if i > 0:
                        new_prev = (word[i - 1], word[i])
                        stats[new_prev] += freq
                        indices[new_prev].add(idx)
                    
                    if i < len(word) - 1:
                        new_prev = (word[i], word[i + 1])
                        stats[new_prev] += freq
                        indices[new_prev].add(idx)
                    
                    # 注意：合并后 i 不需要移动，直接进入下一轮 while 即可
                    pass
                else:
                    i += 1
        
        # [TODO 10]: 清理战场，将 best_pair 彻底从 stats 和 indices 中抹除
        if best_pair in stats: del stats[best_pair]
        if best_pair in indices: del indices[best_pair]

    # --- 5. 构建最终的词表 ---
    # [TODO 11]: 按顺序将 merges 里的新 Token 录入 vocab，ID 从 256 开始递增
    for pair in merges:
        new_id = len(vocab)
        vocab[new_id] = pair[0] + pair[1]
        
   
    
    # [TODO 12]: 将 special_tokens 录入 vocab 的最后位置
     # 添加特殊 Token
    for s_tok in special_tokens:
        s_bytes = s_tok.encode("utf-8")
        vocab[len(vocab)] = s_bytes
    
    return vocab, merges


# ==========================================
# 下方为已完成的 IO 工具代码，无需手撕
# ==========================================

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def save_tokenizer_files(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    byte_encoder = bytes_to_unicode()

    json_vocab = {
        k: "".join(byte_encoder[b] for b in v) 
        for k, v in vocab.items()
    }
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=4)
    
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")

def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt" 
    vocab_size = 10000 
    special_tokens = ["<|endoftext|>"]
    output_dir = "data/TinyStoriesV2-GPT4-train"

    print(f"开始训练 BPE 分词器 (目标词表大小: {vocab_size})...")
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    save_tokenizer_files(vocab, merges, output_dir)
    print("训练及保存完成！")

if __name__ == "__main__":
    main()