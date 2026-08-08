from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "analysis" / "top_ladder_decks_2026_08_07.ipynb"


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (trans)",
            "language": "python",
            "name": "trans",
        },
        "language_info": {"name": "python", "version": "3.10"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Pokémon TCG AI Battle：2026-08-07 天梯前排卡组统计\n\n"
            "可复现分析：官方日 replay、完整 60 张牌表、胜负结果与官方英/日卡牌映射。"
        ),
        nbf.v4.new_markdown_cell("## tl;dr"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "from html import escape\n"
            "import csv, json, subprocess\n"
            "from IPython.display import HTML, Markdown, display\n\n"
            "ROOT = Path.cwd()\n"
            "if not (ROOT / 'analysis' / 'top_ladder_decks.py').exists() and (ROOT / 'top_ladder_decks.py').exists():\n"
            "    ROOT = ROOT.parent\n"
            "CONDA_PYTHON = Path('/homes/lzhang/mypath/new/envs/trans/bin/python')\n"
            "OUTPUT_DIR = ROOT / 'analysis' / 'outputs' / 'top_ladder_2026_08_07'\n"
            "subprocess.run([str(CONDA_PYTHON), '-s', 'analysis/top_ladder_decks.py', '--workers', '16'], cwd=ROOT, check=True)\n\n"
            "def load_csv(name):\n"
            "    with (OUTPUT_DIR / name).open(encoding='utf-8-sig', newline='') as handle:\n"
            "        return list(csv.DictReader(handle))\n\n"
            "def show_table(rows, columns, formatters=None, limit=None):\n"
            "    formatters = formatters or {}\n"
            "    data = rows if limit is None else rows[:limit]\n"
            "    head = ''.join(f'<th>{escape(label)}</th>' for _, label in columns)\n"
            "    body = []\n"
            "    for row in data:\n"
            "        cells = []\n"
            "        for key, _ in columns:\n"
            "            value = row.get(key, '')\n"
            "            if key in formatters and value not in ('', None):\n"
            "                value = formatters[key](value)\n"
            "            cells.append(f'<td>{escape(str(value))}</td>')\n"
            "        body.append('<tr>' + ''.join(cells) + '</tr>')\n"
            "    display(HTML('<table><thead><tr>' + head + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table>'))\n\n"
            "pct = lambda value: f'{float(value):.1%}'\n"
            "num = lambda value: f'{int(float(value)):,}'\n"
            "summary = json.loads((OUTPUT_DIR / 'summary.json').read_text(encoding='utf-8'))\n"
            "lines = [\n"
            "    f\"- **主样本覆盖：** {summary['elite_valid_games']:,}/{summary['elite_target_games']:,} 个高分段目标对局可用，共 {summary['elite_deck_appearances']:,} 个牌表出场、{summary['elite_unique_teams']:,} 支队伍。\",\n"
            "    f\"- **高分段口径：** 每日按 `min_score` 排名前 10%，当前门槛为 {summary['elite_min_score_cutoff']:.3f}，保证一局内双方都不低于该分数。\",\n"
            "    f\"- **当前第一卡组家族：** {summary['top_archetype']}，使用率 {summary['top_archetype_usage_share']:.1%}，非镜像胜率 {summary['top_archetype_nonmirror_win_rate']:.1%}（n={summary['top_archetype_nonmirror_appearances']:,}）。\",\n"
            "    f\"- **先手基线：** 高分段中先手胜率 {summary['first_player_win_rate']:.1%}；卡组胜率应与样本量和对局构成一起看。\",\n"
            "]\n"
            "display(Markdown('\\n'.join(lines)))"
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "### Key Assumptions\n\n"
            "- 把“天梯靠前排”定义为每日按 `min_score` 排名前 10% 的对局。`min_score` 是两名玩家中较低者的分数，因此能保证双方都达到门槛；日清单不提供分数到具体玩家的映射。\n"
            "- replay 在开局动作中包含双方完整 60 张卡 ID；最终 `rewards` 提供胜负。只把双方牌表完整且奖励有效的对局纳入主分析。\n"
            "- 卡组家族基于 Pokémon 卡多重集合聚类：对 Pokémon 核心做 IDF 加权 Jaccard，相似度至少 0.55 归为同一家族；Trainer/Energy 的差别作为具体牌表变体。\n"
            "- 胜率的 Wilson 区间是描述性的；同一队伍的重复对局并非独立样本。镜像对局单独排除后计算主胜率。\n"
            "- manifest 时间戳不带时区偏移，本文只按源时间使用，不推断时区。"
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            "quality = load_csv('data_quality.csv')\n"
            "show_table(quality, [\n"
            "    ('date','日期'), ('manifest_games','Manifest 对局'), ('json_files_present','JSON 文件'),\n"
            "    ('parsed_games','解析成功'), ('deck_complete_games','完整牌表'),\n"
            "    ('valid_games','有效对局'), ('valid_game_coverage','有效覆盖率')\n"
            "], {'manifest_games':num,'json_files_present':num,'parsed_games':num,'deck_complete_games':num,'valid_games':num,'valid_game_coverage':pct})\n"
            "display(Markdown(f\"输出目录：`{OUTPUT_DIR}`\"))"
        ),
        nbf.v4.new_markdown_cell(
            "## Results\n\n"
            "下面先看卡组家族的出场率与非镜像胜率，再看与前一日的使用率变化。"
        ),
        nbf.v4.new_code_cell(
            "archetypes = load_csv('archetype_summary.csv')\n"
            "show_table(archetypes, [\n"
            "    ('rank','排名'), ('archetype_label','卡组家族'), ('appearances','出场'),\n"
            "    ('usage_share','使用率'), ('nonmirror_appearances','非镜像 n'),\n"
            "    ('nonmirror_win_rate','非镜像胜率'), ('nonmirror_ci_low','95% CI 下限'),\n"
            "    ('nonmirror_ci_high','95% CI 上限'), ('unique_teams','队伍'), ('exact_deck_variants','完整牌表变体')\n"
            "], {\n"
            "    'rank':num,'appearances':num,'usage_share':pct,'nonmirror_appearances':num,\n"
            "    'nonmirror_win_rate':pct,'nonmirror_ci_low':pct,'nonmirror_ci_high':pct,\n"
            "    'unique_teams':num,'exact_deck_variants':num\n"
            "}, limit=15)"
        ),
        nbf.v4.new_markdown_cell("### 日对日卡组占比变化"),
        nbf.v4.new_code_cell(
            "comparison = load_csv('day_comparison.csv')\n"
            "show_table(comparison, [\n"
            "    ('archetype_label','卡组家族'), ('current_appearances','8/7 出场'),\n"
            "    ('current_usage_share','8/7 使用率'), ('previous_appearances','8/6 出场'),\n"
            "    ('previous_usage_share','8/6 使用率'), ('usage_share_delta_pp','变化 pp')\n"
            "], {\n"
            "    'current_appearances':num,'current_usage_share':pct,'previous_appearances':num,\n"
            "    'previous_usage_share':pct,'usage_share_delta_pp':lambda x:f'{float(x):+.1f}'\n"
            "}, limit=15)"
        ),
        nbf.v4.new_markdown_cell("### 高分段常见卡与相对全场提升"),
        nbf.v4.new_code_cell(
            "usage = load_csv('card_usage.csv')\n"
            "for group in ['Pokémon','Trainer','Energy']:\n"
            "    display(Markdown(f'#### {group}'))\n"
            "    rows = [row for row in usage if row['card_group'] == group][:12]\n"
            "    show_table(rows, [\n"
            "        ('card_id','ID'),('card_name_en','English'),('card_name_jp','日本語'),\n"
            "        ('elite_inclusion_count','纳入牌表'),('elite_inclusion_rate','高分段纳入率'),\n"
            "        ('avg_copies_when_included','纳入时均张数'),('field_inclusion_rate','全场纳入率'),\n"
            "        ('elite_vs_field_lift','Lift')\n"
            "    ], {\n"
            "        'card_id':num,'elite_inclusion_count':num,'elite_inclusion_rate':pct,\n"
            "        'avg_copies_when_included':lambda x:f'{float(x):.2f}',\n"
            "        'field_inclusion_rate':pct,'elite_vs_field_lift':lambda x:f'{float(x):.2f}x'\n"
            "    })"
        ),
        nbf.v4.new_markdown_cell("### 头部卡组对局矩阵"),
        nbf.v4.new_code_cell(
            "matrix_rows = load_csv('matchup_win_rate_matrix.csv')\n"
            "if matrix_rows:\n"
            "    first_key = next(iter(matrix_rows[0]))\n"
            "    matrix_columns = [(first_key,'卡组')] + [(key,key) for key in matrix_rows[0] if key != first_key]\n"
            "    matrix_formatters = {key:pct for key in matrix_rows[0] if key != first_key}\n"
            "    show_table(matrix_rows, matrix_columns, matrix_formatters)\n"
            "display(Markdown('对应样本量保存在 `matchup_sample_size_matrix.csv`；空白或小样本对局不应据此下确定性结论。'))"
        ),
        nbf.v4.new_markdown_cell("### 代表性完整牌表"),
        nbf.v4.new_code_cell(
            "decklists = load_csv('representative_decklists.csv')\n"
            "show_table(decklists, [\n"
            "    ('archetype_rank','家族排名'),('archetype_label','卡组家族'),('card_id','卡 ID'),\n"
            "    ('card_name_en','English'),('card_name_jp','日本語'),('card_group','类别'),('count','张数')\n"
            "], {'archetype_rank':num,'card_id':num,'count':num}, limit=80)\n"
            "display(Markdown('完整前十卡组代表牌表已导出到 `representative_decklists.csv`。'))"
        ),
        nbf.v4.new_markdown_cell("## Takeaways"),
        nbf.v4.new_code_cell(
            "top = archetypes[0]\n"
            "second = archetypes[1]\n"
            "delta = float(comparison[0]['usage_share_delta_pp'])\n"
            "takeaways = [\n"
            "    f\"1. **备战优先级先按出场率排：** `{top['archetype_label']}` 是当前最常见家族（{float(top['usage_share']):.1%}），其次是 `{second['archetype_label']}`（{float(second['usage_share']):.1%}）。\",\n"
            "    f\"2. **胜率只作方向性证据：** 头部家族的非镜像胜率为 {float(top['nonmirror_win_rate']):.1%}，95% Wilson 区间 {float(top['nonmirror_ci_low']):.1%}–{float(top['nonmirror_ci_high']):.1%}；同队重复对局和 matchup mix 会影响估计。\",\n"
            "    f\"3. **关注结构变化：** 第一家族较前一日使用率变化 {delta:+.1f} 个百分点；赛前应继续按同一口径刷新最新日 replay。\",\n"
            "    \"4. **落地建议：** 先针对出场率最高的三类做定向 matchup 回放和起手/先后手切分，再决定是否因小样本高胜率卡组调整构筑。\",\n"
            "]\n"
            "display(Markdown('\\n'.join(takeaways)))"
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
