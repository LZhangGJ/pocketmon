from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
TITLE = "Agent 55328694 全量 Replay 败局审计"
KAGGLE_URL = (
    "https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/"
    "submissions?submissionId=55328694"
)
FLAG_LABELS = {
    "declined_optional_selection": "可选检索选空",
    "ended_with_attach_available": "可贴能量却结束回合",
    "ended_with_evolve_available": "可进化却结束回合",
}


def pct(value: float) -> str:
    return f"{value:.1%}"


def mistake_text(review: dict[str, Any] | None) -> str:
    if review is None:
        return "未发现高置信度单步失误"
    parts = []
    for flag, label in FLAG_LABELS.items():
        if flag not in review["flags"]:
            continue
        turns = ", ".join(f"T{turn}" for turn in review["turns"][flag])
        parts.append(f"{label} ×{review['flags'][flag]}（{turns}）")
    return "；".join(parts)


def main() -> None:
    review = json.loads((OUTPUT / "review_summary.json").read_text(encoding="utf-8"))
    episodes = json.loads((OUTPUT / "episodes.json").read_text(encoding="utf-8"))
    replay_check = json.loads(
        (OUTPUT / "agent_replay_check.json").read_text(encoding="utf-8")
    )
    reviewed_by_id = {
        int(row["episode_id"]): row for row in review["reviewed_losses"]
    }

    headline = [
        {
            "public_games": review["public_games"],
            "public_win_rate": review["win_rate"],
            "affected_losses": review["high_confidence_affected_loss_episodes"],
            "affected_loss_rate": review["high_confidence_affected_loss_rate"],
            "fallback_calls": replay_check["totals"]["fallbackCalls"],
            "mismatch_count": replay_check["totals"]["mismatches"],
            "comparison_count": replay_check["totals"]["comparisons"],
        }
    ]

    flag_comparison = []
    for row in review["flags"]:
        if not row["high_confidence"]:
            continue
        for result, rate_key, episode_key, denominator in (
            ("败局", "loss_episode_rate", "loss_episodes", review["losses"]),
            ("胜局", "win_episode_rate", "win_episodes", review["wins"]),
        ):
            flag_comparison.append(
                {
                    "pattern": row["label"],
                    "result": result,
                    "episode_rate": row[rate_key],
                    "episodes": row[episode_key],
                    "denominator": denominator,
                    "loss_minus_win_pp": row["loss_minus_win_pp"],
                }
            )

    matchup = []
    for rank, row in enumerate(
        sorted(review["matchups"], key=lambda value: (value["win_rate"], -value["games"])),
        start=1,
    ):
        matchup.append({**row, "rank_by_win_rate": rank})

    loss_detail = []
    for row in episodes:
        if "PUBLIC" not in str(row["episode_type"]) or row["result"] != "loss":
            continue
        review_row = reviewed_by_id.get(int(row["episode_id"]))
        loss_detail.append(
            {
                "episode_id": int(row["episode_id"]),
                "severity": review_row["severity"] if review_row else "—",
                "opponent": row["opponent"],
                "archetype": next(
                    (
                        candidate["archetype"]
                        for candidate in review["reviewed_losses"]
                        if int(candidate["episode_id"]) == int(row["episode_id"])
                    ),
                    "其他/未标记",
                ),
                "opponent_deck": row["opponent_deck"],
                "max_turn": row["max_turn"],
                "event_count": review_row["event_count"] if review_row else 0,
                "assessment": mistake_text(review_row),
            }
        )
    loss_detail.sort(key=lambda row: (-row["event_count"], row["episode_id"]))

    database_path = OUTPUT / "report_source.sqlite"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        DROP TABLE IF EXISTS headline_metrics;
        DROP TABLE IF EXISTS flag_comparison;
        DROP TABLE IF EXISTS matchup;
        DROP TABLE IF EXISTS loss_detail;
        CREATE TABLE headline_metrics (
            public_games INTEGER,
            public_win_rate REAL,
            affected_losses INTEGER,
            affected_loss_rate REAL,
            fallback_calls INTEGER,
            mismatch_count INTEGER,
            comparison_count INTEGER
        );
        CREATE TABLE flag_comparison (
            pattern TEXT,
            result TEXT,
            episode_rate REAL,
            episodes INTEGER,
            denominator INTEGER,
            loss_minus_win_pp REAL
        );
        CREATE TABLE matchup (
            archetype TEXT,
            games INTEGER,
            wins INTEGER,
            losses INTEGER,
            win_rate REAL,
            loss_share REAL,
            rank_by_win_rate INTEGER
        );
        CREATE TABLE loss_detail (
            episode_id INTEGER,
            severity TEXT,
            opponent TEXT,
            archetype TEXT,
            opponent_deck TEXT,
            max_turn INTEGER,
            event_count INTEGER,
            assessment TEXT
        );
        """
    )
    connection.executemany(
        "INSERT INTO headline_metrics VALUES (:public_games, :public_win_rate, :affected_losses, :affected_loss_rate, :fallback_calls, :mismatch_count, :comparison_count)",
        headline,
    )
    connection.executemany(
        "INSERT INTO flag_comparison VALUES (:pattern, :result, :episode_rate, :episodes, :denominator, :loss_minus_win_pp)",
        flag_comparison,
    )
    connection.executemany(
        "INSERT INTO matchup VALUES (:archetype, :games, :wins, :losses, :win_rate, :loss_share, :rank_by_win_rate)",
        matchup,
    )
    connection.executemany(
        "INSERT INTO loss_detail VALUES (:episode_id, :severity, :opponent, :archetype, :opponent_deck, :max_turn, :event_count, :assessment)",
        loss_detail,
    )
    connection.commit()
    connection.row_factory = sqlite3.Row

    headline_sql = "SELECT public_games, public_win_rate, affected_losses, affected_loss_rate, fallback_calls, mismatch_count, comparison_count FROM headline_metrics"
    flag_sql = "SELECT pattern, result, episode_rate, episodes, denominator, loss_minus_win_pp FROM flag_comparison ORDER BY pattern, result"
    matchup_sql = "SELECT archetype, games, wins, losses, win_rate, loss_share, rank_by_win_rate FROM matchup ORDER BY win_rate ASC, games DESC"
    loss_sql = "SELECT episode_id, severity, opponent, archetype, opponent_deck, max_turn, event_count, assessment FROM loss_detail ORDER BY event_count DESC, episode_id ASC"

    def query_rows(sql: str) -> list[dict[str, Any]]:
        return [dict(row) for row in connection.execute(sql).fetchall()]

    headline = query_rows(headline_sql)
    flag_comparison = query_rows(flag_sql)
    matchup = query_rows(matchup_sql)
    loss_detail = query_rows(loss_sql)
    connection.close()

    common_source = {
        "path": "research/submission_55328694_replay_audit/output/report_source.sqlite",
        "href": KAGGLE_URL,
    }
    headline_source = {
        **common_source,
        "id": "headline_sql",
        "label": "全量 replay 头部指标与提交包复现结果",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": headline_sql,
            "description": "读取 107 场公开对局的头部结果和原提交包的精确动作复现计数。",
            "executed_at": "2026-08-09T00:00:00+09:00",
            "tables_used": ["headline_metrics"],
            "filters": ["公开胜率排除 90751124 validation 自对局"],
            "metric_definitions": [
                "公开胜率 = 64 / 107",
                "受影响败局 = 43 场败局中至少出现一项高置信度战术失误信号的 episode 数",
                "fallback = 原提交 main.py 捕获异常或产生非法动作后启用最小合法动作的次数",
            ],
        },
    }
    flag_source = {
        **common_source,
        "id": "flag_sql",
        "label": "胜负对局中的动作信号出现率",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": flag_sql,
            "description": "按结果比较三类高置信度动作信号的 episode 覆盖率。",
            "executed_at": "2026-08-09T00:00:00+09:00",
            "tables_used": ["flag_comparison"],
            "filters": ["仅含 107 场 EPISODE_TYPE_PUBLIC"],
            "metric_definitions": ["对局出现率 = 出现该信号的 episode 数 / 对应结果 episode 数"],
        },
    }
    matchup_source = {
        **common_source,
        "id": "matchup_sql",
        "label": "按对手牌组聚合的公开赛结果",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": matchup_sql,
            "description": "按对手主轴牌组聚合 107 场公开对局的胜负。",
            "executed_at": "2026-08-09T00:00:00+09:00",
            "tables_used": ["matchup"],
            "filters": ["仅含 107 场 EPISODE_TYPE_PUBLIC"],
            "metric_definitions": ["牌组胜率 = 胜局数 / 对局数；小样本只作方向性参考"],
        },
    }
    loss_source = {
        **common_source,
        "id": "loss_sql",
        "label": "43 场公开败局逐局审计",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": loss_sql,
            "description": "列出全部公开败局及规则审计识别到的高置信度战术失误。",
            "executed_at": "2026-08-09T00:00:00+09:00",
            "tables_used": ["loss_detail"],
            "filters": ["result = loss", "episode_type = EPISODE_TYPE_PUBLIC"],
            "metric_definitions": ["事件数 = 该 episode 中三类高置信度战术失误的出现次数之和"],
        },
    }

    manifest = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": "对 Kaggle submission 55328694 的 108 场 episode 做逐动作败局审计。",
        "generatedAt": "2026-08-09T00:00:00+09:00",
        "sources": [headline_source, flag_source, matchup_source, loss_source],
        "cards": [
            {
                "id": "public_games_card",
                "description": "公开赛 episode 数；validation 自对局不计入。",
                "dataset": "headline",
                "sourceId": "headline_sql",
                "metrics": [{"label": "公开对局", "field": "public_games", "format": "number"}],
            },
            {
                "id": "win_rate_card",
                "description": "64 胜、43 负。",
                "dataset": "headline",
                "sourceId": "headline_sql",
                "metrics": [{"label": "公开胜率", "field": "public_win_rate", "format": "percent"}],
            },
            {
                "id": "affected_losses_card",
                "description": "至少出现一项高置信度战术失误信号的败局。",
                "dataset": "headline",
                "sourceId": "headline_sql",
                "metrics": [
                    {"label": "受影响败局", "field": "affected_losses", "format": "number"},
                    {"label": "占全部败局", "field": "affected_loss_rate", "format": "percent"},
                ],
            },
            {
                "id": "fallback_card",
                "description": "原提交代码精确复现全部动作后的运行计数。",
                "dataset": "headline",
                "sourceId": "headline_sql",
                "metrics": [
                    {"label": "Fallback", "field": "fallback_calls", "format": "number"},
                    {"label": "动作不匹配", "field": "mismatch_count", "format": "number"},
                    {"label": "对照动作", "field": "comparison_count", "format": "compact"},
                ],
            },
        ],
        "charts": [
            {
                "id": "flag_comparison_chart",
                "title": "三类动作信号在胜局与败局中的对局出现率",
                "subtitle": "任一高置信度信号在败局中覆盖 51.2%，胜局中为 31.2%；单项差异以检索选空最大。",
                "type": "bar",
                "dataset": "flag_comparison",
                "sourceId": "flag_sql",
                "valueFormat": "percent",
                "encodings": {
                    "x": {"field": "pattern", "type": "nominal", "label": "动作信号"},
                    "y": {"field": "episode_rate", "type": "quantitative", "label": "对局出现率", "format": "percent"},
                    "color": {"field": "result", "type": "nominal", "label": "结果"},
                    "tooltip": [
                        {"field": "episodes", "type": "quantitative", "label": "出现对局数"},
                        {"field": "denominator", "type": "quantitative", "label": "结果对局数"},
                        {"field": "loss_minus_win_pp", "type": "quantitative", "label": "败局-胜局", "unit": "pp"},
                    ],
                },
            },
            {
                "id": "matchup_chart",
                "title": "公开对局按对手牌组聚合的胜率",
                "subtitle": "Mega Lopunny 为 2-5，Grass / Ogerpon 为 3-4；这两组样本较小，只宜用作定向回归测试。",
                "type": "bar",
                "dataset": "matchup",
                "sourceId": "matchup_sql",
                "valueFormat": "percent",
                "encodings": {
                    "x": {"field": "archetype", "type": "nominal", "label": "对手牌组"},
                    "y": {"field": "win_rate", "type": "quantitative", "label": "胜率", "format": "percent"},
                    "tooltip": [
                        {"field": "games", "type": "quantitative", "label": "对局数"},
                        {"field": "wins", "type": "quantitative", "label": "胜局"},
                        {"field": "losses", "type": "quantitative", "label": "败局"},
                        {"field": "loss_share", "type": "quantitative", "label": "败局占比", "format": "percent"},
                    ],
                },
            },
        ],
        "tables": [
            {
                "id": "loss_audit_table",
                "title": "43 场公开败局逐局审计",
                "subtitle": "先列出事件较多的对局；“未发现”表示本规则集没有识别到明确的单步资源浪费，不代表整局最优。",
                "dataset": "loss_detail",
                "sourceId": "loss_sql",
                "density": "compact",
                "defaultSort": {"field": "event_count", "direction": "desc"},
                "columns": [
                    {"field": "episode_id", "label": "Episode", "type": "number"},
                    {"field": "severity", "label": "严重度", "type": "text"},
                    {"field": "archetype", "label": "对手类型", "type": "text"},
                    {"field": "max_turn", "label": "末回合", "type": "number"},
                    {"field": "event_count", "label": "事件数", "type": "number"},
                    {"field": "assessment", "label": "审计结论", "type": "text"},
                ],
            }
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
            {
                "id": "executive_summary",
                "type": "markdown",
                "body": (
                    "## Executive Summary\n\n"
                    "- **完整范围是 108 场 episode。** 其中 107 场公开对局为 64 胜 43 负（59.8%）；90751124 是 validation 自对局，双方都是同一 submission。\n"
                    "- **最确定的失误不是非法动作，而是主动放弃价值。** 43 场败局中，22 场至少出现一次“检索选空 / 能贴不贴 / 能进化不进化”；合计 37 个事件。\n"
                    "- **这些信号与败局相关，但不是每次都致败。** 任一信号在败局覆盖 51.2%，胜局覆盖 31.2%；仍有 21 场败局没有识别到明确的单步失误。\n"
                    "- **代码异常不是主因。** 用原提交包重放 10,546 个动作，动作完全一致，fallback 与不匹配均为 0。"
                ),
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": [
                    "public_games_card",
                    "win_rate_card",
                    "affected_losses_card",
                    "fallback_card",
                ],
            },
            {
                "id": "mistake_patterns",
                "type": "markdown",
                "body": (
                    "## 我们确实犯了三类重复失误\n\n"
                    "**第一是可选检索直接选空：15 次，分布在 11 场败局。** 典型情形是已打出 Buddy-Buddy Poffin，系统给出 Snorunt 或 Impidimp，却返回空选择。这个信号的败局覆盖率为 25.6%，比胜局高 14.6 个百分点，是三类中最稳定的异常。\n\n"
                    "**第二是仍可贴能量却结束回合：10 次，分布在 7 场败局。** 其中 90761788、90771647、91103678、91221601、91307030 在第 1 回合就空过贴能；90761788 还在第 7、9 回合重复。\n\n"
                    "**第三是仍可进化却结束回合：12 次，分布在 7 场败局。** 这会推迟 Grimmsnarl ex 的成型、血量与攻击窗口。与“攻击时仍有能力/手牌可用”不同，这三类信号在败局里更常见，因此优先级更高。"
                ),
            },
            {"id": "flag_comparison", "type": "chart", "chartId": "flag_comparison_chart"},
            {
                "id": "episode_91325674",
                "type": "markdown",
                "body": (
                    "## 91325674 是最典型的崩盘样本\n\n"
                    "**第 2 回合，agent 打出第二张 Buddy-Buddy Poffin 后，在 Snorunt、Impidimp、Snorunt 三个可选目标前返回空选择。** 随后它在第 6、8、10、12 回合连续四次持有并可合法进化 Grimmsnarl ex，却直接结束回合；直到第 14 回合才通过 Rare Candy 建成第一只 Grimmsnarl ex。\n\n"
                    "这不是一次偶然排序错误，而是同一局连续四个决策点重复选择 END，说明主阶段价值排序缺少最低限度的资源推进约束。"
                ),
            },
            {
                "id": "matchup_findings",
                "type": "markdown",
                "body": (
                    "## 败局还集中在两个弱对局\n\n"
                    "**Mega Lopunny 对局只有 2 胜 5 负（28.6%），Grass / Ogerpon 为 3 胜 4 负（42.9%）。** 这两组样本不大，但明显弱于总体 59.8%，适合作为下一轮定向回归集。\n\n"
                    "Grimmsnarl 镜像贡献 17 场败局、Alakazam 贡献 9 场，合计占全部败局 60.5%；不过它们也占了绝大多数对局，胜率分别为 61.4% 和 59.1%，更像高暴露量而非结构性劣势。"
                ),
            },
            {"id": "matchup_visual", "type": "chart", "chartId": "matchup_chart"},
            {
                "id": "all_losses_intro",
                "type": "markdown",
                "body": (
                    "## 43 场败局逐局审计\n\n"
                    "下表覆盖全部公开败局。22 场带有高置信度单步失误；其余 21 场没有发现“合法但明显放弃当回合资源”的动作，更多需要靠反事实模拟判断累计路线、换位和攻击目标是否次优。"
                ),
            },
            {"id": "all_losses", "type": "table", "tableId": "loss_audit_table"},
            {
                "id": "recommendations",
                "type": "markdown",
                "body": (
                    "## 推荐修改顺序\n\n"
                    "1. **先加主阶段 guardrail。** 若模型选择 END，但本回合尚未贴能且存在合法 ATTACH，优先贴给可成型的 Grimmsnarl 线；若存在合法 Grimmsnarl ex 进化，除明确的反制规则外禁止直接 END。\n"
                    "2. **修 optional-count head。** 对 Buddy-Buddy Poffin 这类已支付卡牌成本且有空板凳的检索，把可选数量下限由模型自由选 0 改为规则约束至少 1，并按 Snorunt / Impidimp 的盘面需求排序。当前实现先独立预测数量，再取 option logits 的 top-k，正是“好目标存在但数量头选 0”的结构性入口。\n"
                    "3. **用 22 场败局做 hard-negative 微调。** 给 END 对比合法 ATTACH / EVOLVE、给空选择对比非空检索，增加同盘面成对样本；验收指标直接用这三类事件率，而不只看整体 action accuracy。\n"
                    "4. **单独扩充 Mega Lopunny 与 Grass / Ogerpon 回归集。** 先把每类至少补到 30 场，再决定是否需要对手牌组条件化策略或换牌。"
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## 还需要回答的问题\n\n"
                    "- 把 37 个动作逐一替换为 guardrail 动作后，实际可挽回多少场？这需要可复现的环境状态或分支模拟，replay 本身不能证明因果。\n"
                    "- 第二张 Poffin 选空时，是否应优先补 Snorunt 还是 Impidimp？应按当前板凳、进化件、奖赏区与对局类型制定目标排序。\n"
                    "- 对手牌组辅助头在 9,559 次模型调用中只有 101 次置信度超过 60%，而预测只用于统计；是否值得将其校准后真正接入策略路由？"
                ),
            },
            {
                "id": "caveats",
                "type": "markdown",
                "body": (
                    "## 边界与假设\n\n"
                    "本审计使用 2026 年 8 月 9 日取得的完整 episode 快照。公开统计只含 107 场 public episode；90751124 validation 自对局单独检查，败方在第 4 回合跳过贴能，并在第 10 回合两次放弃 Snorunt 可选检索。\n\n"
                    "“高置信度失误”表示 replay 中存在合法且通常增值的动作，但 agent 选择了空检索或 END；它不等于该动作一定改变胜负。相反，“攻击时仍有手牌/能力/进化可用”在胜局更常见，已从失误清单剔除，避免把合理资源保留误判为错误。"
                ),
            },
        ],
    }

    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": "2026-08-09T00:00:00+09:00",
            "status": "ready",
            "datasets": {
                "headline": headline,
                "flag_comparison": flag_comparison,
                "matchup": matchup,
                "loss_detail": loss_detail,
            },
        },
        "sources": [headline_source, flag_source, matchup_source, loss_source],
    }
    (OUTPUT / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "title": TITLE,
                "blocks": len(manifest["blocks"]),
                "datasets": {key: len(rows) for key, rows in artifact["snapshot"]["datasets"].items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
