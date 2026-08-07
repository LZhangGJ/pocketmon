#!/usr/bin/env python3
"""Build the bounded Data Analytics report payload for the Kaggle audit."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INTEL = ROOT / "research" / "kaggle_intelligence" / "2026-08-06"
TITLE = "PTCG AI Battle 金牌冲刺情报与行动方案"


def display_group(rank: int, method: str) -> str:
    if rank == 3:
        return "学习模型混合系统"
    if rank in {1, 12}:
        return "Meta/组合构建器"
    if rank in {2, 4, 6, 8, 10, 13, 14, 16, 19}:
        return "规则 + 受控搜索"
    return "规则/专家策略"


def main() -> None:
    audit = json.loads((INTEL / "analysis" / "notebook_audit.json").read_text(encoding="utf-8"))
    score_rows = []
    method_rows = []
    for row in audit["notebooks"]:
        rank = int(row["rank"])
        owner = str(row["ref"]).split("/", 1)[0]
        short_title = str(row["title"])
        if len(short_title) > 34:
            short_title = short_title[:31] + "…"
        item = {
            "rank": rank,
            "notebook": short_title,
            "owner": owner,
            "score": float(row["score"]),
            "votes": int(row["votes"]),
            "method_group": display_group(rank, str(row["reviewed_method"])),
            "method": str(row["reviewed_method"]),
            "lineage": str(row["lineage"]),
            "review_note": str(row["review_note"]),
            "url": str(row["url"]),
        }
        score_rows.append(item)
        method_rows.append(item)

    notebook_query = (
        "SELECT rank, CASE WHEN length(title) > 34 THEN substr(title, 1, 31) || '…' ELSE title END AS notebook, "
        "split_part(ref, '/', 1) AS owner, score, votes, reviewed_method AS method, lineage, review_note, url, "
        "CASE WHEN rank = 3 THEN '学习模型混合系统' "
        "WHEN rank IN (1, 12) THEN 'Meta/组合构建器' "
        "WHEN rank IN (2, 4, 6, 8, 10, 13, 14, 16, 19) THEN '规则 + 受控搜索' "
        "ELSE '规则/专家策略' END AS method_group "
        "FROM read_csv_auto('research/kaggle_intelligence/2026-08-06/analysis/notebook_audit.csv') ORDER BY rank"
    )

    sources = [
        {
            "id": "competition_rules",
            "label": "Kaggle competition overview and evaluation",
            "href": "https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview/description",
        },
        {
            "id": "notebook_audit",
            "label": "Top-20 notebook source audit, observed 2026-08-06",
            "path": "research/kaggle_intelligence/2026-08-06/analysis/notebook_audit.json",
            "query": {
                "id": "top20_notebook_audit_20260806",
                "description": "Select the reviewed top-20 notebook ranking, methods, lineages, notes, and display groups.",
                "language": "sql",
                "engine": "duckdb",
                "sql": notebook_query,
                "tables_used": ["research/kaggle_intelligence/2026-08-06/analysis/notebook_audit.csv"],
                "executed_at": "2026-08-06T15:30:00+09:00",
                "metric_definitions": [
                    "score = Kaggle Code Public Score observed on 2026-08-06; dynamic and not a causal method estimate.",
                    "rank = descending Public Score order observed on the competition Code page on 2026-08-06.",
                ],
            },
        },
        {
            "id": "discussion_archive",
            "label": "Downloaded Kaggle discussion archive: 202 topics / 1,011 messages",
            "path": "research/kaggle_intelligence/2026-08-06/discussions/index.csv",
        },
        {
            "id": "repo_baseline",
            "label": "Current repository BC and evaluation baseline",
            "path": "docs/RL_BC_002_PLAN.md",
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {
            "id": "executive_summary",
            "type": "markdown",
            "body": (
                "## Executive Summary\n\n"
                "- 前 20 个公开高分 Notebook 折叠为约 10 个方法谱系；没有可证明的纯 RL/纯 BC 赢家。唯一明显含学习模型的强条目，仍依赖 expert router 和大量规则 guard。\n"
                "- 最后十天应采用双槽混合策略：槽 A 追求当前 top-band meta 加权胜率，槽 B 选择不同牌组和失败模式做反制。\n"
                "- BC 当前不应替换规则教师；最佳短期用途是 confidence-gated candidate ranker/residual。搜索只在高杠杆节点启用，并要求可靠叶价值、margin gate 和完整 fallback。\n"
                "- Public Score 只做外部校准。候选晋级由 paired-seat、固定对手 hash、Wilson/Bootstrap 区间、最差 matchup 和零功能失败共同决定。"
            ),
        },
        {
            "id": "public_methods",
            "type": "markdown",
            "body": (
                "## 公开高分方法的真正共识\n\n"
                "公开强方法的共同结构是：**牌组降低决策难度 → 专用规则/专家基线 → 对局路由 → 关键节点受控搜索或模型建议 → 窄 guard → 合法 fallback**。"
                "排名 4/8/14 为完全相同源码，排名 10/16 相似度 95.7%，说明 Code 排名前 20 并非 20 个独立证据。"
            ),
        },
        {"id": "score_chart_block", "type": "chart", "chartId": "notebook_scores"},
        {
            "id": "method_table_intro",
            "type": "markdown",
            "body": (
                "## 方法审计明细\n\n"
                "表中方法来自源码、内嵌压缩资产和 Notebook 叙述的人工复核。分数为 2026-08-06 动态快照，不作为方法因果效果。"
            ),
        },
        {"id": "method_table_block", "type": "table", "tableId": "method_table"},
        {
            "id": "failure_lessons",
            "type": "markdown",
            "body": (
                "## Discussion 暴露的三类陷阱\n\n"
                "1. **搜索会放大错误价值。** 社区复盘中，beam 给弱策略带来 +11.3pp，却让强策略下降约 15pp。\n"
                "2. **BC 准确率不等于 on-policy 强度。** 72%–79% validation accuracy 的自报案例仍可能只有约 600–700 ELO；小错误会把策略带入教师未覆盖状态。\n"
                "3. **线上 rating 噪声很大。** 最新讨论记录同一 Agent 两次提交约 400 分差。官方从 μ=600 初始化并动态匹配，短期分数不适合做细粒度 hill-climb。"
            ),
        },
        {
            "id": "recommendation",
            "type": "markdown",
            "body": (
                "## 建议：双槽混合制胜\n\n"
                "**槽 A（Meta-weighted champion）：** 在 Grimmsnarl hybrid、Alakazam bounded-search specialist 和现有 Lucario champion 中，用最新 replay payoff matrix 选加权胜率最高者。\n\n"
                "**槽 B（Decorrelated counter）：** 使用不同牌组与失败模式。若 A 是 Grimmsnarl/Alakazam，优先验证 Dragapult + Crushing Hammer 或 Garchomp；若复杂 pilot 不可靠，则回退到成熟 Alakazam/Metal 规则 Agent。\n\n"
                "**技术边界：** 规则负责合法性和灾难 guard；BC 负责候选排序/残差；搜索只在 MAIN context、有限分支、0.2–0.8 秒预算和足够 override margin 下运行。"
            ),
        },
        {
            "id": "ten_day_plan",
            "type": "markdown",
            "body": (
                "## 未来十天的金牌冲刺\n\n"
                "- **8/6–8/8：** 把 10 个公开谱系做成哈希冻结 opponent pool；用最新 replay 重建 top-band deck clusters 与 payoff matrix。\n"
                "- **8/8–8/10：** 实现 BC confidence gate，并完成一个 Grim hybrid 或 Alakazam bounded-search challenger。\n"
                "- **8/10–8/12：** 每 matchup 200–400 局 paired-seat 快筛，先淘汰失败率、最差 matchup 或尾部风险不合格者。\n"
                "- **8/12–8/14：** 前 2–3 个候选扩大到约 1,000+ 局/关键 matchup，冻结代码和对手 hash。\n"
                "- **8/14–8/16：** 保留线上 champion，用另一个槽测试 challenger；最终上传两个互补且已冻结的 Agent，停止高风险改动。"
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "- 最新 replay 的 top-band Grimmsnarl/Alakazam/Garchomp/Dragapult 份额和互胜矩阵是什么？\n"
                "- BC 在高置信覆盖 10%/25%/50% 决策时，override 的净胜率提升分别是多少？\n"
                "- Search API 在最终 runtime 中是否稳定、进程隔离后是否仍有正收益？\n"
                "- 两个最终槽是否存在共同 hard counter，需要牺牲均值换低尾部风险？"
            ),
        },
        {
            "id": "caveats",
            "type": "markdown",
            "body": (
                "## Caveats and Assumptions\n\n"
                "- Notebook 分数、票数和排序是 2026-08-06 快照；高分条目可能是 builder、复制版或过期 meta 报告。\n"
                "- 6 月 29 日 Notebook 和 7 月 26 日 meta 已有时效风险，最终牌组必须由最新 replay 重验。\n"
                "- Discussion 的 rating、胜率和方法分类含自报/推断成分；本报告已尽量用源码和仓库证据交叉验证。\n"
                "- 聊天中暴露的 Kaggle Token 未被使用或保存。"
            ),
        },
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": "Kaggle public discussion and top-notebook audit with a ten-day gold-medal strategy.",
        "generatedAt": "2026-08-06T15:30:00+09:00",
        "sources": sources,
        "charts": [
            {
                "id": "notebook_scores",
                "title": "公开高分 Notebook 排名快照",
                "subtitle": "20 个条目包含重复/近复制家族，分数差不能直接解释为方法差异。",
                "type": "bar",
                "dataset": "notebook_scores",
                "sourceId": "notebook_audit",
                "encodings": {
                    "x": {"field": "notebook", "type": "nominal", "title": "Notebook"},
                    "y": {"field": "score", "type": "quantitative", "title": "Public Score"},
                    "color": {"field": "method_group", "type": "nominal", "title": "方法组"},
                },
                "options": {"orientation": "horizontal", "grouping": "grouped"},
            }
        ],
        "tables": [
            {
                "id": "method_table",
                "title": "Top-20 源码审计与方法谱系",
                "dataset": "method_table",
                "sourceId": "notebook_audit",
                "columns": [
                    {"field": "rank", "label": "排名", "type": "number"},
                    {"field": "score", "label": "分数", "type": "number"},
                    {"field": "owner", "label": "作者", "type": "text"},
                    {"field": "notebook", "label": "Notebook", "type": "text"},
                    {"field": "method_group", "label": "方法组", "type": "text"},
                    {"field": "method", "label": "复核方法", "type": "text"},
                    {"field": "lineage", "label": "谱系", "type": "text"},
                    {"field": "review_note", "label": "审计备注", "type": "text"},
                ],
                "defaultSort": {"field": "rank", "direction": "asc"},
            }
        ],
        "blocks": blocks,
    }
    snapshot = {
        "version": 1,
        "status": "ready",
        "generatedAt": "2026-08-06T15:30:00+09:00",
        "datasets": {"notebook_scores": score_rows, "method_table": method_rows},
    }
    output = {"manifest": manifest, "snapshot": snapshot, "sources": sources, "surface": "report"}
    target = INTEL / "report_artifact.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
