"""Pokémon TCG AI Battle — 対戦リプレイ観戦ビューア (watch_game)

cabt エンジンで 1 試合を最後まで進め、各決定点での **選んだ行動** と、ターン頭の
**盤面スナップショット**（両者のアクティブ/ベンチの HP・エネ、サイド残数、手札）を
日本語カード名で記録し、ブラウザで 1 手ずつ観戦できる **単一 HTML ビューア** を出力する。
プレイを目で追って改善案を探すための道具。

cg のバトル obs はイベントログ(logs)が空のため、行動は select 内容からデコードし、
ダメージ・KO はサイド残数と HP の差分から読み取る。

────────────────────────────────────────────────────────────────────
このノートブックの動かし方（Kaggle）
────────────────────────────────────────────────────────────────────
1. 右の "Add Input" から **コンペ "pokemon-tcg-ai-battle"** を追加する
   （cg ネイティブライブラリ libcg.so をここから取得する。再配布はしない）。
2. そのまま Run。`/kaggle/working/replay.html` が出力される
   → Output タブからダウンロードし、ブラウザで開くと ◀▶ / 再生 / スライダー /
     矢印キーで盤面・手札を 1 手ずつ観戦できる（単一 HTML・外部依存なし）。

※ デモの対戦は「合法手からランダムに選ぶ」軽量な仮エージェント同士。
   自作エージェントの戦略コードは含めていない（公開しない）。実エージェントで
   観戦したい場合は first_legal/random の代わりに自分の agent(obs)->list[int] を差し込む。

このファイルは元リポジトリ src/watch_game.py を Kaggle 単体実行用に整えたもの。
"""

from __future__ import annotations

import csv
import glob
import json
import random
import sys
from pathlib import Path

# ── 1. cg ライブラリを Kaggle の input から探して import 可能にする ──────────────
def _find_cg_parent() -> Path:
    """libcg.so を含む 'cg' パッケージの親ディレクトリを /kaggle/input から探す。"""
    # すでに import 可能ならそれを使う
    try:
        import cg  # noqa: F401
        return Path(cg.__file__).resolve().parent.parent
    except Exception:
        pass
    for so in glob.glob("/kaggle/input/**/cg/libcg.so", recursive=True):
        return Path(so).resolve().parent.parent  # .../<parent>/cg/libcg.so → <parent>
    raise RuntimeError(
        "cg ライブラリ(libcg.so)が見つかりません。右の Add Input から "
        "コンペ 'pokemon-tcg-ai-battle' を追加してください。"
    )


_CG_PARENT = _find_cg_parent()
if str(_CG_PARENT) not in sys.path:
    sys.path.insert(0, str(_CG_PARENT))

from cg.api import (  # noqa: E402
    AreaType,
    OptionType,
    SelectContext,
    all_attack,
    all_card_data,
)
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402


# ── 2. 日本語カード名の読み込み（任意。無ければ cg の英名で代替） ─────────────────
def _find_jp_csv() -> Path | None:
    for p in glob.glob("/kaggle/input/**/JP_Card_Data.csv", recursive=True):
        return Path(p)
    return None


def _load_jp_names() -> dict[int, str]:
    names: dict[int, str] = {}
    jp = _find_jp_csv()
    if jp is not None and jp.is_file():
        with open(jp, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if len(row) >= 2 and row[0].strip().isdigit():
                    names.setdefault(int(row[0]), row[1].strip())
    for c in all_card_data():  # cg の英名で補完
        names.setdefault(c.cardId, c.name)
    return names


NAMES: dict[int, str] = {}
ATTACKS: dict[int, tuple[str, int]] = {}


def nm(cid: int | None) -> str:
    if cid is None:
        return "?"
    return NAMES.get(cid, f"#{cid}")


# ── 3. obs → HTML ビューア用フレーム変換（src/watch_game.py より移植） ─────────────
def fmt_mon(p: dict | None) -> str:
    if not p:
        return "なし"
    s = f"{nm(p['id'])}(HP{p['hp']}/{p['maxHp']} エネ{len(p.get('energies') or [])}"
    tools = p.get("tools") or []
    if tools:
        s += f" 道具{len(tools)}"
    return s + ")"


def _mon_json(p: dict | None) -> dict | None:
    if not p:
        return None
    return {
        "name": nm(p["id"]),
        "hp": p.get("hp", 0),
        "maxHp": p.get("maxHp", 0),
        "energies": list(p.get("energies") or []),
        "tools": len(p.get("tools") or []),
    }


def _hand_cards(pl: dict) -> list[str] | None:
    """手札が公開(手番プレイヤー自身)ならカード名一覧、非公開なら None（枚数のみ）。"""
    hand = pl.get("hand")
    if hand is None:
        return None
    out = []
    for c in hand:
        try:
            out.append(nm(c["id"]))
        except (KeyError, TypeError):
            out.append("?")
    return out


def _side_json(pl: dict) -> dict:
    act = pl.get("active") or []
    return {
        "active": _mon_json(act[0] if act else None),
        "bench": [_mon_json(b) for b in (pl.get("bench") or []) if b],
        "prize": len(pl.get("prize") or []),
        "deck": pl.get("deckCount", 0),
        "hand": pl.get("handCount", 0),
        "handCards": _hand_cards(pl),
        "status": [
            s
            for s, on in (
                ("どく", pl.get("poisoned")),
                ("やけど", pl.get("burned")),
                ("ねむり", pl.get("asleep")),
                ("マヒ", pl.get("paralyzed")),
                ("こんらん", pl.get("confused")),
            )
            if on
        ],
    }


def _frame_json(obs: dict, us: int, action: str) -> dict:
    cur = obs["current"]
    stadium = cur.get("stadium") or []
    return {
        "turn": cur["turn"],
        "tp": cur["yourIndex"],
        "us": us,
        "action": action,
        "stadium": nm(stadium[0]["id"]) if stadium else None,
        "you": _side_json(cur["players"][us]),
        "opp": _side_json(cur["players"][1 - us]),
    }


def board(cur: dict, us: int) -> str:
    lines = []
    for who, idx in (("自分", us), ("相手", 1 - us)):
        pl = cur["players"][idx]
        act = pl.get("active") or [None]
        active = act[0] if act else None
        bench = [fmt_mon(b) for b in (pl.get("bench") or []) if b]
        prize = len(pl.get("prize") or [])
        lines.append(
            f"  {who}(P{idx}): 場={fmt_mon(active)} / "
            f"ベンチ[{', '.join(bench) if bench else '－'}] / サイド残{prize}"
        )
    return "\n".join(lines)


def _card_in(obs: dict, area: int | None, index: int | None, pidx: int) -> int | None:
    if area is None or index is None:
        return None
    cur = obs["current"]
    pl = cur["players"][pidx]
    sel = obs.get("select") or {}
    try:
        if area == int(AreaType.HAND):
            return (pl.get("hand") or [])[index]["id"]
        if area == int(AreaType.ACTIVE):
            return (pl.get("active") or [])[index]["id"]
        if area == int(AreaType.BENCH):
            return (pl.get("bench") or [])[index]["id"]
        if area == int(AreaType.DISCARD):
            return (pl.get("discard") or [])[index]["id"]
        if area == int(AreaType.DECK):
            return (sel.get("deck") or [])[index]["id"]
        if area == int(AreaType.STADIUM):
            return (cur.get("stadium") or [])[index]["id"]
        if area == int(AreaType.LOOKING):
            return (cur.get("looking") or [])[index]["id"]
    except (IndexError, KeyError, TypeError):
        return None
    return None


def fmt_action(obs: dict, o: dict, tp: int) -> str:
    cur = obs["current"]
    t = o.get("type")
    hand = cur["players"][tp].get("hand") or []

    def hand_name(i):
        try:
            return nm(hand[i]["id"])
        except (IndexError, KeyError, TypeError):
            return "?"

    def inplay_name():
        cid = _card_in(obs, o.get("inPlayArea"), o.get("inPlayIndex"), tp)
        return nm(cid)

    if t == int(OptionType.PLAY):
        return f"プレイ: {hand_name(o.get('index'))}"
    if t == int(OptionType.ATTACH):
        return f"装着: {hand_name(o.get('index'))} → {inplay_name()}"
    if t == int(OptionType.EVOLVE):
        src = inplay_name()
        dst = o.get("cardId")
        return f"進化: {src}" + (f" → {nm(dst)}" if dst is not None else " を進化")
    if t == int(OptionType.ABILITY):
        cid = _card_in(obs, o.get("area"), o.get("index"), tp)
        return f"特性: {nm(cid)}"
    if t == int(OptionType.RETREAT):
        return "にげる"
    if t == int(OptionType.ATTACK):
        name, dmg = ATTACKS.get(o.get("attackId"), (f"attackId={o.get('attackId')}", 0))
        return f"ワザ: {name}（基本{dmg}）"
    if t == int(OptionType.END):
        return "ターン終了"
    if t == int(OptionType.NUMBER):
        return f"数値: {o.get('number')}"
    if t == int(OptionType.YES):
        return "はい"
    if t == int(OptionType.NO):
        return "いいえ"
    if t in (
        int(OptionType.CARD),
        int(OptionType.TOOL_CARD),
        int(OptionType.ENERGY_CARD),
        int(OptionType.ENERGY),
    ):
        cid = _card_in(obs, o.get("area"), o.get("index"), o.get("playerIndex", tp))
        return f"カード選択: {nm(cid)}"
    if t == int(OptionType.DISCARD):
        cid = _card_in(obs, o.get("area"), o.get("index"), tp)
        return f"トラッシュ: {nm(cid)}"
    return f"type={t}"


CTX = {
    int(SelectContext.MAIN): "メイン",
    int(SelectContext.SETUP_ACTIVE_POKEMON): "初期アクティブ選択",
    int(SelectContext.SETUP_BENCH_POKEMON): "初期ベンチ選択",
    int(SelectContext.SWITCH): "入れ替え先選択",
    int(SelectContext.TO_ACTIVE): "アクティブへ出す",
    int(SelectContext.TO_BENCH): "ベンチへ出す",
    int(SelectContext.TO_HAND): "手札に加える(サーチ)",
    int(SelectContext.DISCARD): "トラッシュ選択",
    int(SelectContext.TO_PRIZE): "サイドへ",
    int(SelectContext.DAMAGE_COUNTER): "ダメカン配置",
    int(SelectContext.DAMAGE_COUNTER_ANY): "ダメカン自由配置",
    int(SelectContext.DAMAGE): "ダメージ対象選択",
    int(SelectContext.HEAL): "回復対象選択",
    int(SelectContext.ATTACH_FROM): "エネ/道具の付け先",
    int(SelectContext.EFFECT_TARGET): "効果対象選択",
    int(SelectContext.ATTACK): "ワザ選択",
    int(SelectContext.EVOLVE): "進化",
    int(SelectContext.DRAW_COUNT): "ドロー枚数",
    int(SelectContext.DAMAGE_COUNTER_COUNT): "ダメカン数",
    int(SelectContext.IS_FIRST): "先攻希望?",
    int(SelectContext.MULLIGAN): "マリガン?",
    int(SelectContext.ACTIVATE): "効果を使う?",
    int(SelectContext.COIN_HEAD): "コイン表を選ぶ?",
}


# ── 4. HTML ビューア（src/watch_game.py の単一HTMLテンプレートをそのまま使用） ─────
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>対戦ビューア — __TITLE__</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, "Hiragino Sans", "Noto Sans JP", sans-serif;
         background:#0f1320; color:#e8ecf4; }
  header { padding:10px 16px; background:#171c2e; border-bottom:1px solid #2a3350; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  #caption { padding:8px 16px; background:#1d2440; font-size:14px; min-height:20px;
             border-bottom:1px solid #2a3350; }
  #caption .ko { color:#ffd24d; font-weight:700; }
  #board { display:flex; flex-direction:column; gap:8px; padding:14px; max-width:1000px; margin:0 auto; }
  .side { border:1px solid #2a3350; border-radius:10px; padding:10px; }
  .side.opp { background:#241a24; }
  .side.you { background:#16233a; }
  .side .label { font-size:12px; opacity:.7; margin-bottom:6px; display:flex; justify-content:space-between; }
  .row { display:flex; gap:8px; flex-wrap:wrap; align-items:flex-end; }
  .active-wrap { display:flex; gap:8px; align-items:flex-end; }
  .mon { width:118px; background:#0d1322; border:1px solid #33406a; border-radius:8px;
         padding:6px; font-size:11px; position:relative; }
  .mon.active { border-color:#ffd24d; box-shadow:0 0 0 1px #ffd24d55; width:132px; }
  .mon .nm { font-weight:700; font-size:11px; line-height:1.25; height:28px; overflow:hidden; }
  .hpbar { height:7px; background:#2a3350; border-radius:4px; overflow:hidden; margin:4px 0 2px; }
  .hpbar > div { height:100%; }
  .hptxt { font-variant-numeric:tabular-nums; opacity:.85; }
  .pips { display:flex; gap:2px; flex-wrap:wrap; margin-top:4px; min-height:10px; }
  .pip { width:10px; height:10px; border-radius:50%; border:1px solid #00000040; }
  .tool { position:absolute; top:5px; right:5px; font-size:9px; background:#7a5; color:#000;
          border-radius:3px; padding:0 3px; }
  .st { margin-top:3px; }
  .st span { font-size:9px; background:#a33; border-radius:3px; padding:0 3px; margin-right:2px; }
  .meta { font-size:11px; opacity:.8; display:flex; gap:10px; flex-wrap:wrap; }
  .hand { margin-top:8px; border-top:1px dashed #2a3350; padding-top:6px; }
  .hand .ttl { font-size:11px; opacity:.7; margin-bottom:4px; }
  .chips { display:flex; gap:4px; flex-wrap:wrap; }
  .chip { font-size:10px; background:#222a44; border:1px solid #44507a; border-radius:10px; padding:1px 8px; white-space:nowrap; }
  .chip b { color:#ffd24d; }
  .hand.hidden .ttl { opacity:.5; font-style:italic; }
  .prizebar { display:inline-flex; gap:2px; vertical-align:middle; }
  .prizebar i { width:7px; height:10px; background:#ffd24d; border-radius:1px; display:inline-block; }
  .prizebar i.taken { background:#2a3350; }
  .empty { opacity:.4; font-style:italic; padding:8px; }
  #controls { position:sticky; bottom:0; background:#171c2e; border-top:1px solid #2a3350;
              padding:10px 16px; display:flex; gap:10px; align-items:center; }
  #controls button { background:#2a3350; color:#e8ecf4;
                     border:1px solid #44507a; border-radius:6px; padding:6px 12px; cursor:pointer; font-size:14px; }
  #controls button:hover { background:#34406a; }
  #slider { flex:1; }
  #counter { font-variant-numeric:tabular-nums; font-size:13px; min-width:120px; text-align:right; }
  .stadium { font-size:11px; opacity:.85; }
</style>
</head>
<body>
<header><h1>🃏 対戦ビューア — __TITLE__</h1></header>
<div id="caption"></div>
<div id="board"></div>
<div id="controls">
  <button id="first">⏮</button>
  <button id="prev">◀ 前</button>
  <button id="play">▶ 再生</button>
  <button id="next">次 ▶</button>
  <button id="last">⏭</button>
  <input id="slider" type="range" min="0" value="0">
  <span id="counter"></span>
</div>
<script>
const FRAMES = __FRAMES__;
const ECOLOR = {0:"#c8c8c8",1:"#6abe30",2:"#e8503a",3:"#3aa0e8",4:"#f0c000",5:"#b060c0",6:"#c07030",7:"#404858",8:"#9aa6b2",9:"#b89020",10:"#9ad3c8",11:"#804060"};
let idx = 0, timer = null;

function hpColor(r){ return r>0.5 ? "#46c266" : r>0.25 ? "#e8c34a" : "#e8503a"; }

function monHTML(m, active){
  if(!m) return '<div class="mon'+(active?' active':'')+'"><div class="empty">なし</div></div>';
  const r = m.maxHp ? Math.max(0, m.hp/m.maxHp) : 0;
  let pips = (m.energies||[]).map(e=>'<span class="pip" style="background:'+(ECOLOR[e]||"#888")+'"></span>').join('');
  let tool = m.tools>0 ? '<span class="tool">道具'+m.tools+'</span>' : '';
  let st = (m.status&&m.status.length)?'<div class="st">'+m.status.map(s=>'<span>'+s+'</span>').join('')+'</div>':'';
  return '<div class="mon'+(active?' active':'')+'">'+tool+
    '<div class="nm">'+m.name+'</div>'+
    '<div class="hpbar"><div style="width:'+(r*100)+'%;background:'+hpColor(r)+'"></div></div>'+
    '<div class="hptxt">HP '+m.hp+'/'+m.maxHp+'</div>'+
    '<div class="pips">'+pips+'</div>'+st+'</div>';
}
function prizeBar(n){ let s='<span class="prizebar">'; for(let i=0;i<6;i++) s+='<i class="'+(i<n?'':'taken')+'"></i>'; return s+'</span>'; }
function handHTML(s){
  if(s.handCards === null || s.handCards === undefined)
    return '<div class="hand hidden"><div class="ttl">手札 '+s.hand+'枚（非公開）</div></div>';
  const m = {}; s.handCards.forEach(n=>{ m[n]=(m[n]||0)+1; });
  const chips = Object.keys(m).map(n=>'<span class="chip">'+n+(m[n]>1?' <b>×'+m[n]+'</b>':'')+'</span>').join('');
  return '<div class="hand"><div class="ttl">手札 '+s.hand+'枚</div><div class="chips">'+(chips||'<span class="chip">なし</span>')+'</div></div>';
}
function sideHTML(s, cls, who){
  let bench = (s.bench&&s.bench.length)? s.bench.map(b=>monHTML(b,false)).join('') : '<div class="empty">ベンチなし</div>';
  return '<div class="side '+cls+'">'+
    '<div class="label"><span>'+who+'</span><span class="meta">サイド'+prizeBar(s.prize)+' ・ 手札'+s.hand+' ・ 山札'+s.deck+'</span></div>'+
    '<div class="active-wrap"><div>バトル場</div>'+monHTML(s.active,true)+'<div style="margin-left:12px">ベンチ</div><div class="row">'+bench+'</div></div>'+
    handHTML(s)+
    '</div>';
}
function render(){
  const f = FRAMES[idx];
  const youTurn = (f.tp === f.us);
  let cap = '試合'+f.game+' ／ ターン'+f.turn+' ／ 手番 '+(youTurn?'★P'+f.tp:'P'+f.tp)+'　';
  let act = f.action || '';
  if(act.indexOf('試合終了')>=0 || act.indexOf('サイド')>=0) act = '<span class="ko">'+act+'</span>';
  document.getElementById('caption').innerHTML = cap + '<b>'+(youTurn?'★ ':'')+'</b>' + act;
  const stadium = f.stadium ? '<div class="stadium">スタジアム: '+f.stadium+'</div>' : '';
  document.getElementById('board').innerHTML =
    sideHTML(f.opp,'opp','P'+(1-f.us)) + stadium + sideHTML(f.you,'you','P'+f.us);
  document.getElementById('counter').textContent = (idx+1)+' / '+FRAMES.length;
  document.getElementById('slider').value = idx;
}
function go(i){ idx = Math.max(0, Math.min(FRAMES.length-1, i)); render(); }
function play(){
  if(timer){ clearInterval(timer); timer=null; document.getElementById('play').textContent='▶ 再生'; return; }
  document.getElementById('play').textContent='⏸ 停止';
  timer = setInterval(()=>{ if(idx>=FRAMES.length-1){ clearInterval(timer); timer=null; document.getElementById('play').textContent='▶ 再生'; } else go(idx+1); }, 900);
}
document.getElementById('first').onclick=()=>go(0);
document.getElementById('prev').onclick=()=>go(idx-1);
document.getElementById('next').onclick=()=>go(idx+1);
document.getElementById('last').onclick=()=>go(FRAMES.length-1);
document.getElementById('play').onclick=play;
document.getElementById('slider').max=FRAMES.length-1;
document.getElementById('slider').oninput=e=>go(+e.target.value);
document.addEventListener('keydown',e=>{ if(e.key==='ArrowRight')go(idx+1); else if(e.key==='ArrowLeft')go(idx-1); });
render();
</script>
</body>
</html>
"""


def write_html(frames: list[dict], title: str, path: Path) -> None:
    if not frames:
        empty = {"active": None, "bench": [], "prize": 6, "deck": 0, "hand": 0, "status": []}
        frames = [{"turn": 0, "tp": 0, "us": 0, "action": "（フレームなし）",
                   "stadium": None, "you": empty, "opp": empty, "game": 1}]
    data = json.dumps(frames, ensure_ascii=False)
    html = _HTML_TEMPLATE.replace("__FRAMES__", data).replace("__TITLE__", title)
    path.write_text(html, encoding="utf-8")


# ── 5. デモ用の軽量エージェント（合法手からランダム）と公開サンプルデッキ ──────────
# デッキはコンペ公開の sample_submission のもの（60枚, 戦略を含まない）。
DEMO_DECK = (
    [1158, 721, 721, 722, 722, 722, 722, 723, 723, 723, 723,
     1145, 1145, 1145, 1145, 1205, 1205, 1227, 1227, 1227, 1227,
     1235, 1235, 1235, 1235]
    + [3] * 35
)


def make_random_agent(seed: int):
    rng = random.Random(seed)

    def agent(obs: dict) -> list[int]:
        sel = obs.get("select")
        if sel is None:  # 初期選択: デッキ(カードID 60枚)を返す
            return list(DEMO_DECK)
        n = len(sel.get("option") or [])
        if n == 0:
            return []
        lo = sel.get("minCount") or 0
        hi = sel.get("maxCount") or 1
        k = min(hi, n)
        if k < lo:
            k = min(lo, n)
        k = max(k, 1)
        idxs = list(range(n))
        rng.shuffle(idxs)
        return sorted(idxs[:k])

    return agent


# ── 6. 1 試合を記録してフレーム列を作る（src/watch_game.py の watch() を移植） ──────
def record_game(agent0, deck0, agent1, deck1, us: int = 0) -> tuple[list[dict], str]:
    global NAMES, ATTACKS
    NAMES = _load_jp_names()
    ATTACKS = {a.attackId: (a.name, a.damage) for a in all_attack()}

    agents = (agent0, agent1)
    frames: list[dict] = []
    obs, start = battle_start(deck0, deck1)
    if start.errorPlayer >= 0:
        battle_finish()
        raise ValueError(f"デッキエラー (player={start.errorPlayer}, type={start.errorType})")

    try:
        while obs["current"]["result"] < 0:
            cur = obs["current"]
            tp = cur["yourIndex"]
            sel = obs.get("select")
            selected = agents[tp](obs)

            action_text = ""
            if sel is not None:
                opts = sel.get("option") or []
                ctx = CTX.get(sel.get("context"), f"ctx{sel.get('context')}")
                acts = [fmt_action(obs, opts[i], tp) for i in selected[:3] if 0 <= i < len(opts)]
                if acts:
                    action_text = f"{ctx}: {' / '.join(acts)}"
            if action_text:
                fr = _frame_json(obs, us, action_text)
                fr["game"] = 1
                frames.append(fr)
            obs = battle_select(selected)
    finally:
        battle_finish()

    res = obs["current"]["result"]
    winner = "P{}の勝ち🎉".format(us) if res == us else ("引き分け" if res == 2 else "P{}の勝ち".format(1 - us))
    fin = obs["current"]
    end = _frame_json(obs, us, f"【試合終了】{winner}")
    end["game"] = 1
    frames.append(end)
    summary = (
        f"{winner}（P{us}サイド残{len(fin['players'][us].get('prize') or [])} / "
        f"P{1 - us}サイド残{len(fin['players'][1 - us].get('prize') or [])}, {fin['turn']}ターン）"
    )
    return frames, summary


# ── 7. 実行: デモ 1 試合を記録 → replay.html を出力 ─────────────────────────────
def main() -> None:
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
    a0 = make_random_agent(0)
    a1 = make_random_agent(1)
    frames, summary = record_game(a0, list(DEMO_DECK), a1, list(DEMO_DECK), us=0)
    title = "ランダム仮エージェント同士のデモ対戦"
    html_path = out_dir / "replay.html"
    write_html(frames, title, html_path)
    print(f"✓ リプレイHTMLを書き出しました: {html_path}（{len(frames)}フレーム）")
    print(f"  結果: {summary}")
    print("  → Output タブから replay.html をダウンロードし、ブラウザで開いてください")
    print("    （◀▶ / 再生 / スライダー / 矢印キーで 1 手ずつ観戦できます）")

    # Notebook 環境ならインラインでも表示を試みる
    try:
        from IPython.display import HTML, display  # noqa: PLC0415
        display(HTML(html_path.read_text(encoding="utf-8")))
    except Exception:
        pass


if __name__ == "__main__":
    main()
