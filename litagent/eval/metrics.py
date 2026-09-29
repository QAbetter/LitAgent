"""Phase 4：检索质量指标。

输入是「检索返回的页码列表（有序）」与「预期命中的页码集合」，
输出 Hit Rate / Recall@k / MRR 等指标，用于对比不同检索器的效果。
"""


def hit_at_k(returned_pages: list[int], gt_pages: set[int], k: int) -> float:
    """Top-k 里是否命中任一预期页，命中返回 1.0，否则 0.0。"""
    return 1.0 if (set(returned_pages[:k]) & set(gt_pages)) else 0.0


def recall_at_k(returned_pages: list[int], gt_pages: set[int], k: int) -> float:
    """Top-k 命中的预期页数占比（去重后），对多页 ground truth 更公平。"""
    if not gt_pages:
        return 0.0
    return len(set(returned_pages[:k]) & set(gt_pages)) / len(gt_pages)


def reciprocal_rank(returned_pages: list[int], gt_pages: set[int], k: int) -> float:
    """首个命中的倒数排名（MRR 单条），没命中返回 0.0。"""
    for i, page in enumerate(returned_pages[:k], start=1):
        if page in gt_pages:
            return 1.0 / i
    return 0.0


def evaluate(returned_pages: list[int], gt_pages: set[int], k: int) -> dict:
    """单条样本的三个指标。"""
    return {
        "hit": hit_at_k(returned_pages, gt_pages, k),
        "recall": recall_at_k(returned_pages, gt_pages, k),
        "mrr": reciprocal_rank(returned_pages, gt_pages, k),
    }


def summarize(per_query: list[dict]) -> dict:
    """对多条样本的指标求平均，得到整体评价值。"""
    if not per_query:
        return {"hit": 0.0, "recall": 0.0, "mrr": 0.0}
    keys = ("hit", "recall", "mrr")
    return {key: sum(q[key] for q in per_query) / len(per_query) for key in keys}