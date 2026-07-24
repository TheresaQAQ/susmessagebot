"""
Evaluate classify_message on gold-labeled game-community samples.

Usage:
  python -m scripts.eval_accuracy --prompt-version v2_zh_balanced --run 1
  python -m scripts.eval_accuracy --prompt-version v2_zh_balanced --run 1 --limit 20
  python -m scripts.eval_accuracy --prompt-version v2_zh_balanced --run 1 --resume
  python -m scripts.eval_accuracy --prompt-version v2_zh_balanced --run 1 --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from susmessagebot.prompt_loader import DEFAULT_PROMPT_ID, list_prompt_ids

GOLD: list[tuple[str, str, str]] = [
    # ---- BAN: ads / promo / selling (50) ----
    ("稳定代练全服可接，上分包月私我，价格优惠加V详谈", "BAN", "zh-ad"),
    ("收徒教打金，日入过千，有意的来私聊", "BAN", "zh-ad"),
    ("卖号出毕业号，支持中介，价格可刀，加微信", "BAN", "zh-ad"),
    ("低价金币大量现货，秒发，群里别问直接私我", "BAN", "zh-ad"),
    ("陪玩上门服务，年轻小姐姐，详情私聊", "BAN", "zh-ad"),
    ("拉人进我们公会 Discord，有福利红包，链接私发", "BAN", "zh-ad"),
    ("工作室代肝周本，包金包材料，价目表私信领取", "BAN", "zh-ad"),
    ("出号：满配账号只要999，先到先得私我", "BAN", "zh-ad"),
    ("专业破解辅助出售，不死不封号，感兴趣DM", "BAN", "zh-ad"),
    ("招募代理卖CDK，高返利，加QQ详谈", "BAN", "zh-ad"),
    ("Cheap Mythic+ boosts and raid carries, DM rates", "BAN", "en-ad"),
    ("Selling legendary mounts and gold, discord.gg/fakepromo", "BAN", "en-ad"),
    ("Buy my leveling service 1-80, paypal ready, message me", "BAN", "en-ad"),
    ("Join our premium carry discord for weekly clears", "BAN", "en-ad"),
    ("Account shop open: full gear characters for sale", "BAN", "en-ad"),
    ("Undetected cheats for sale, lifetime key DM me", "BAN", "en-ad"),
    ("Looking for clients for paid coaching, $20/hr paypal", "BAN", "en-ad"),
    ("Referral code XYZ999 get free crystals, use mine and DM", "BAN", "en-ad"),
    ("We buy your accounts cash, high price, WhatsApp only", "BAN", "en-ad"),
    ("Powerleveling agency hiring boosters AND taking orders", "BAN", "en-ad"),
    ("代行あります。割引中。DMください。他鯖招待もあります", "BAN", "ja-ad"),
    ("アカウント販売します。値下げ交渉OK、DMへ", "BAN", "ja-ad"),
    ("대리육성 받습니다. 가격 문의 DM", "BAN", "ko-ad"),
    ("Продаю золото и прокачку, пишите в ЛС", "BAN", "ru-ad"),
    ("Vendo cuentas y carries, escribe por DM", "BAN", "es-ad"),
    ("g u a r a n t e e d r e t u r n s on sol? ping me quietly", "BAN", "en-evasion"),
    ("fr33 g0ld & b00sts, d m me n0w", "BAN", "en-evasion"),
    ("代 练 包 月 私 我，价 格 优 惠", "BAN", "zh-evasion"),
    ("加我QQ领免费抽奖，中奖率99%", "BAN", "zh-scam"),
    ("Earn $5000/week playing games from home, DM details", "BAN", "en-scam"),
    ("Customs fee stuck, need 280 SGD PayNow, refund in 1h", "BAN", "en-scam"),
    ("Market research panel pays daily, comment IN to join", "BAN", "en-scam"),
    ("今晚有空私我，有稳的旁白单，先交小保证金锁名额", "BAN", "zh-scam"),
    ("Our logistics firm hiring remote coords, onboarding fee refundable", "BAN", "en-scam"),
    ("出售游戏内充值代付，费率超低，放心冲", "BAN", "zh-ad"),
    ("出租高战力账号，时租日租都行私聊", "BAN", "zh-ad"),
    ("工作室开荒团队收费位，门票私聊老板", "BAN", "zh-ad"),
    ("Hacked clients cheap, install guide after payment", "BAN", "en-ad"),
    ("discord.gg/boost4u weekly mythic sales", "BAN", "en-ad"),
    ("出初音周边联名礼盒，群友价私我下单", "BAN", "zh-ad"),
    ("Coin flip betting group, 2x money fast, invite only DM", "BAN", "en-scam"),
    ("代练不是广告哦仅交流，上分私聊价格单", "BAN", "zh-ad"),
    ("不是卖号，是帮朋友转让，有意私", "BAN", "zh-ad"),
    ("Free Nitro giveaway verify here: http://totally-legit-nitro.xyz", "BAN", "en-scam"),
    ("工作室找固定队，但是要先交团费进语音", "BAN", "zh-ad"),
    ("Selling raid loot funnel service, gold or cash", "BAN", "en-ad"),
    ("收米收金高价，公群勿扰私聊报价", "BAN", "zh-ad"),
    ("Vietnamese gold seller, inbox me for rate card", "BAN", "en-ad"),
    ("陪玩语音甜，按小时计费，加V安排", "BAN", "zh-ad"),
    ("RMT shop open 24/7, telegram @FakeGoldSeller", "BAN", "en-ad"),
    # ---- BAN: cross-platform diversion / off-platform funnel ----
    ("内部福利群只在微信，扫码进群领体验金，返利高", "BAN", "zh-divert"),
    ("加QQ群88392011领新区开服红包，群里有价目表", "BAN", "zh-divert"),
    ("稳赚盘只走Telegram，搜 @WinClub88 进群看单", "BAN", "zh-divert"),
    ("WhatsApp me +65 9xxx for gold rate card, same day delivery", "BAN", "en-divert"),
    ("LINEで代行受付中、スタンプ送ってください。料金表あります", "BAN", "ja-divert"),
    ("카톡 아이디 주시면 대리 단가표 보내드려요", "BAN", "ko-divert"),
    ("抖音私信「上车」发你下载链接，充100送100别错过", "BAN", "zh-divert"),
    ("小红书搜同款笔记私信我，带你进返利社群", "BAN", "zh-divert"),
    ("加我Steam好友出毕业号，聊好再交易，群里别公开", "BAN", "zh-divert"),
    ("IG DM @boost4u_official for mythic carries, payment first", "BAN", "en-divert"),
    ("闲鱼店铺搜「星耀代肝」，下单后加旺旺转微信详谈", "BAN", "zh-divert"),
    ("Join our Facebook group for weekly giveaways, admin will DM invite link", "BAN", "en-divert"),
    # ---- SAFE: normal game chat / LFG / discussion ----
    ("今晚有人一起打本吗，缺一个奶", "SAFE", "zh-lfg"),
    ("有没有人教我这个副本机制，老是灭团", "SAFE", "zh-chat"),
    ("新版本这个装备怎么选词条比较好", "SAFE", "zh-chat"),
    ("刚才那波BOSS好难啊，再开一趟吗", "SAFE", "zh-chat"),
    ("有人组队爬塔吗，我输出位", "SAFE", "zh-lfg"),
    ("服务器又卡了，官方啥时候修", "SAFE", "zh-chat"),
    ("今天掉了把武器，好开心", "SAFE", "zh-chat"),
    ("谁知道世界BOSS刷新时间", "SAFE", "zh-chat"),
    ("又有人在群里打代练广告，别信别私聊", "SAFE", "zh-meta"),
    ("代练广告能不能管管，烦死了", "SAFE", "zh-meta"),
    ("anyone down for a dungeon later, need a healer", "SAFE", "en-lfg"),
    ("how do you dodge the second phase slam?", "SAFE", "en-chat"),
    ("patch notes look rough for tanks ngl", "SAFE", "en-chat"),
    ("LF2M heroic, chill run no rush", "SAFE", "en-lfg"),
    ("my fps tanks in the new map, any settings tips?", "SAFE", "en-chat"),
    ("bro these boost ads are getting out of hand", "SAFE", "en-meta"),
    ("congrats on the clear, that was clean", "SAFE", "en-chat"),
    ("is the weekly reset at 8 UTC or 9?", "SAFE", "en-chat"),
    ("looking for a static for next season, casual vibes", "SAFE", "en-lfg"),
    ("don't click fake nitro links people", "SAFE", "en-meta"),
    ("今夜誰かマルチしませんか？ヒーラーいます", "SAFE", "ja-lfg"),
    ("このボスのギミック誰か教えて", "SAFE", "ja-chat"),
    ("오늘 저녁 파티 구해요 딜러입니다", "SAFE", "ko-lfg"),
    ("Кто в пати на вечер? Нужен танк", "SAFE", "ru-lfg"),
    ("¿Alguien para mazmorra? Soy DPS", "SAFE", "es-lfg"),
    ("这个活动任务第3步卡了，求指点", "SAFE", "zh-chat"),
    ("感觉这赛季平衡性有点崩", "SAFE", "zh-chat"),
    ("有人观战吗我准备打排位", "SAFE", "zh-chat"),
    ("刚入坑，萌新求带一局友好点的", "SAFE", "zh-lfg"),
    ("语音里好吵我先切文字了", "SAFE", "zh-chat"),
    ("lol that wipe was my bad, misclicked ultimate", "SAFE", "en-chat"),
    ("what weapon are you running for speed clears?", "SAFE", "en-chat"),
    ("server maintenance again? classic", "SAFE", "en-chat"),
    ("gg wp, see you next raid night", "SAFE", "en-chat"),
    ("anyone else getting rubberbanding near the harbor?", "SAFE", "en-chat"),
    ("SPF warned about fake customs fee scams again, ignore PayNow strangers", "SAFE", "en-meta"),
    ("if anyone is stuck on the quest, wiki page is better than random tutors", "SAFE", "en-meta"),
    ("Hiring a part-time React intern for our open-source docs site, unpaid trial then stipend. Portfolio required. Not crypto.", "SAFE", "en-border"),
    ("congrats you have been selected to receive one free braincell /j", "SAFE", "en-joke"),
    ("bro these earn 5k/week from phone scams getting crazier, don't click DMs", "SAFE", "en-meta"),
    ("今天天气不错，打完本去吃烧烤吗", "SAFE", "zh-chat"),
    ("我把攻略发群文件了，自己看别再被收费坑", "SAFE", "zh-meta"),
    ("排位连跪，心态崩了喝口水", "SAFE", "zh-chat"),
    ("这个皮肤值不值得买啊，纠结", "SAFE", "zh-chat"),
    ("谁假死了害我被炸，下次注意走位", "SAFE", "zh-chat"),
    ("need one more for achievement run, free, no carry fee", "SAFE", "en-lfg"),
    ("can someone explain the new crafting system simply?", "SAFE", "en-chat"),
    ("我不是打广告，就是问今晚本还有位置吗", "SAFE", "zh-lfg"),
    ("哪位大佬有空讲解一下天赋页", "SAFE", "zh-chat"),
    ("语音频道3缺人开荒，自带药", "SAFE", "zh-lfg"),
    # ---- SAFE: mention platforms without diversion intent ----
    ("官方微信公众号刚推了维护公告", "SAFE", "zh-chat"),
    ("有人把攻略传QQ群文件了吗，我找不到", "SAFE", "zh-chat"),
    ("Steam愿望单这皮肤终于打折了", "SAFE", "zh-chat"),
    ("抖音上看到的打法挺有意思，你们试了吗", "SAFE", "zh-chat"),
    ("这篇小红书配装靠谱吗，求打分", "SAFE", "zh-chat"),
    ("Telegram官方频道发了停服通知", "SAFE", "zh-chat"),
    ("LINEの通話よりDiscordの方が聞きやすい", "SAFE", "ja-chat"),
    ("카톡 친구가 오늘 저녁에 온다고 함", "SAFE", "ko-chat"),
    ("bilibili这个up讲解机制讲得挺清楚", "SAFE", "zh-chat"),
    ("闲鱼别买号，容易被骗号被找回", "SAFE", "zh-meta"),
    ("Facebook marketplace listing for a used headset looks sketchy, skip it", "SAFE", "en-meta"),
    ("Instagram filter made my screenshot look cursed lol", "SAFE", "en-chat"),
]

_N_BAN = sum(1 for _, y, _ in GOLD if y == "BAN")
_N_SAFE = sum(1 for _, y, _ in GOLD if y == "SAFE")
assert _N_BAN == _N_SAFE, (_N_BAN, _N_SAFE)
assert len(GOLD) == _N_BAN + _N_SAFE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "eval_results"
RESULTS_DIR = RESULTS_ROOT  # set in main()
RESULTS_PATH = PROJECT_ROOT / "eval_results.jsonl"  # set in main()


def run_dir(prompt_id: str, run: int) -> Path:
    return RESULTS_ROOT / f"prompt_{prompt_id}" / f"run{run}"


def next_run_number(prompt_id: str) -> int:
    base = RESULTS_ROOT / f"prompt_{prompt_id}"
    if not base.exists():
        return 1
    nums = []
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("run"):
            try:
                nums.append(int(p.name[3:]))
            except ValueError:
                pass
    return (max(nums) + 1) if nums else 1


def load_done() -> dict[int, dict]:
    done: dict[int, dict] = {}
    if not RESULTS_PATH.exists():
        return done
    with RESULTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done[row["index"]] = row
    return done


def append_result(row: dict) -> None:
    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def classify_with_retry(text: str, retries: int = 5) -> str:
    from susmessagebot.moderator import classify_message

    delay = 5
    for attempt in range(retries):
        try:
            return classify_message(text)
        except Exception as e:
            msg = str(e)
            logging.warning("classify failed attempt=%s err=%s", attempt + 1, e)
            if "free-models-per-day" in msg or ("Rate limit exceeded" in msg and "per-day" in msg):
                raise RuntimeError(
                    "Free daily quota exhausted. Wait for reset or add credits, then rerun with --resume."
                ) from e
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("unreachable")


def summarize(rows: list[dict], prompt_id: str, run: int, model: str) -> None:
    total = len(rows)
    correct = sum(1 for r in rows if r["predicted"] == r["expected"])
    tp = sum(1 for r in rows if r["expected"] == "BAN" and r["predicted"] == "BAN")
    tn = sum(1 for r in rows if r["expected"] == "SAFE" and r["predicted"] == "SAFE")
    fp = sum(1 for r in rows if r["expected"] == "SAFE" and r["predicted"] == "BAN")
    fn = sum(1 for r in rows if r["expected"] == "BAN" and r["predicted"] == "SAFE")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print("\n========== EVAL SUMMARY ==========")
    print(f"Prompt / Run      : {prompt_id} / run{run}")
    print(f"Model             : {model}")
    print(f"Samples evaluated : {total}/{len(GOLD)}")
    print(f"Accuracy          : {correct}/{total} = {correct/total:.1%}")
    print(f"Confusion         : TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"BAN precision     : {precision:.1%}")
    print(f"BAN recall        : {recall:.1%}")
    print(f"BAN F1            : {f1:.1%}")

    by_tag = Counter()
    by_tag_ok = Counter()
    for r in rows:
        tag = r["tag"].split("-")[0]
        by_tag[tag] += 1
        if r["predicted"] == r["expected"]:
            by_tag_ok[tag] += 1
    print("\nBy language/prefix:")
    for tag in sorted(by_tag):
        print(f"  {tag:6} {by_tag_ok[tag]}/{by_tag[tag]} = {by_tag_ok[tag]/by_tag[tag]:.1%}")

    mistakes = [r for r in rows if r["predicted"] != r["expected"]]
    if mistakes:
        print(f"\nMistakes ({len(mistakes)}):")
        for r in mistakes[:30]:
            print(f"  [{r['index']:03}] expect={r['expected']} pred={r['predicted']} tag={r['tag']} | {r['text'][:60]}")
        if len(mistakes) > 30:
            print(f"  ... and {len(mistakes) - 30} more")
    print("==================================\n")


def _slug(model: str) -> str:
    return model.replace("/", "_")


def save_summary(rows: list[dict], model: str, prompt_id: str, run: int) -> Path:
    total = len(rows)
    correct = sum(1 for r in rows if r["ok"])
    tp = sum(1 for r in rows if r["expected"] == "BAN" and r["predicted"] == "BAN")
    tn = sum(1 for r in rows if r["expected"] == "SAFE" and r["predicted"] == "SAFE")
    fp = sum(1 for r in rows if r["expected"] == "SAFE" and r["predicted"] == "BAN")
    fn = sum(1 for r in rows if r["expected"] == "BAN" and r["predicted"] == "SAFE")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    avg_s = sum(r.get("seconds", 0) for r in rows) / total if total else 0.0
    summary = {
        "prompt_version": prompt_id,
        "run": run,
        "model": model,
        "provider": "siliconflow",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "n": total,
        "accuracy": round(correct / total, 4) if total else 0,
        "correct": correct,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "ban_precision": round(precision, 4),
        "ban_recall": round(recall, 4),
        "ban_f1": round(f1, 4),
        "avg_seconds": round(avg_s, 3),
    }
    path = RESULTS_DIR / f"{_slug(model)}.summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def rebuild_comparison_md() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(RESULTS_DIR.glob("*.summary.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    if not rows:
        return
    prompt_id = rows[0].get("prompt_version", "?")
    run = rows[0].get("run", "?")
    lines = [
        f"# 评测对比 — prompt `{prompt_id}` / run{run}",
        "",
        f"评测集：`eval_accuracy.py` 内置 {len(GOLD)} 条（BAN {_N_BAN} / SAFE {_N_SAFE}，游戏社区反广告，多语言）",
        "",
        "| 模型 | 正确率 | BAN精确率 | BAN召回 | BAN F1 | FP误杀 | FN漏放 | 均耗时 | 记录时间 |",
        "|------|--------|-----------|---------|--------|--------|--------|--------|----------|",
    ]
    for s in sorted(rows, key=lambda x: (-(x.get("accuracy") or 0), x.get("model") or "")):
        lines.append(
            f"| `{s['model']}` | {s['accuracy']:.1%} | {s['ban_precision']:.1%} | "
            f"{s['ban_recall']:.1%} | {s['ban_f1']:.1%} | {s['fp']} | {s['fn']} | "
            f"{s.get('avg_seconds', 0):.2f}s | {s.get('evaluated_at', '')} |"
        )
    lines.append("")
    (RESULTS_DIR / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def rebuild_index_md() -> None:
    """Scan all prompt_*/run* summaries into a master INDEX.md."""
    lines = [
        "# 评测产出索引",
        "",
        "目录约定：`eval_results/prompt_<prompt_id>/run<N>/`",
        "",
        "| 提示词版本 | Run | 模型 | 正确率 | FP | FN | BAN F1 | 均耗时 | 路径 |",
        "|------------|-----|------|--------|----|----|--------|--------|------|",
    ]
    entries = []
    for summary_path in sorted(RESULTS_ROOT.glob("prompt_*/run*/*.summary.json")):
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        prompt_id = s.get("prompt_version") or summary_path.parts[-3].removeprefix("prompt_")
        run = s.get("run") or int(summary_path.parts[-2].removeprefix("run"))
        rel = summary_path.parent.relative_to(PROJECT_ROOT).as_posix()
        entries.append((prompt_id, run, s, rel))

    for prompt_id, run, s, rel in sorted(
        entries, key=lambda x: (x[0], x[1], -(x[2].get("accuracy") or 0), x[2].get("model") or "")
    ):
        lines.append(
            f"| `{prompt_id}` | run{run} | `{s['model']}` | {s['accuracy']:.1%} | "
            f"{s['fp']} | {s['fn']} | {s['ban_f1']:.1%} | {s.get('avg_seconds', 0):.2f}s | `{rel}/` |"
        )

    lines += [
        "",
        "## 提示词文件",
        "",
        "- `susmessagebot/prompts/v1_en_aggressive.md` — 英文偏严（v1 bakeoff）",
        "- `susmessagebot/prompts/v2_zh_balanced.md` — 中文平衡，默认 SAFE，降误杀（当前默认）",
        "",
        "## 探测备注",
        "",
        "- `Qwen/Qwen3.5-4B`：多次调用超时，跳过完整评测",
        "- `THUDM/glm-4-9b-chat` / 部分旧模型：403，账号不可用",
        "- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`：不遵守“只回 SAFE/BAN”，不适合本任务",
        "- bakeoff 默认含 Qwen2.5-7B（v1 弱、v2 可到 98%，应用来对照提示词收益）",
        "",
    ]
    (RESULTS_ROOT / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global RESULTS_DIR, RESULTS_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max samples to run this session (0 = all)")
    parser.add_argument("--resume", action="store_true", help="Skip indexes already in results file")
    parser.add_argument("--sleep", type=float, default=0.3, help="Sleep seconds between calls")
    parser.add_argument("--model", type=str, default="", help="Override SILICONFLOW_MODEL for this run")
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=DEFAULT_PROMPT_ID,
        help=f"Prompt id under susmessagebot/prompts/ (available: {', '.join(list_prompt_ids())})",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=0,
        help="Run number under eval_results/prompt_<id>/runN (0 = auto next)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from susmessagebot import config, moderator

    prompt_id = args.prompt_version
    if prompt_id not in list_prompt_ids():
        raise SystemExit(f"Unknown prompt version: {prompt_id}. Available: {list_prompt_ids()}")

    run = args.run if args.run > 0 else next_run_number(prompt_id)
    os.environ["PROMPT_ID"] = prompt_id
    moderator.PROMPT_ID = prompt_id

    model = args.model or config.SILICONFLOW_MODEL
    if args.model:
        config.SILICONFLOW_MODEL = args.model
        moderator.SILICONFLOW_MODEL = args.model

    RESULTS_DIR = run_dir(prompt_id, run)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH = RESULTS_DIR / f"{_slug(model)}.jsonl"

    print(f"Prompt: {prompt_id}", flush=True)
    print(f"Run:    run{run}", flush=True)
    print(f"Model:  {model}", flush=True)
    print(f"Out:    {RESULTS_DIR}/", flush=True)

    done = load_done() if args.resume else {}
    if not args.resume and RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    limit = args.limit if args.limit > 0 else len(GOLD)
    n_total = len(GOLD)
    rows: list[dict] = []
    ran = 0
    for idx, (text, expected, tag) in enumerate(GOLD):
        if args.resume and idx in done:
            rows.append(done[idx])
            continue
        if ran >= limit:
            break

        print(f"[{idx+1:03}/{n_total}] classifying... | {text[:48]}", flush=True)
        t0 = time.time()
        predicted = classify_with_retry(text)
        elapsed = time.time() - t0
        row = {
            "index": idx,
            "text": text,
            "expected": expected,
            "predicted": predicted,
            "tag": tag,
            "ok": predicted == expected,
            "seconds": round(elapsed, 2),
            "model": model,
            "prompt_version": prompt_id,
            "run": run,
        }
        append_result(row)
        rows.append(row)
        ran += 1
        mark = "OK" if row["ok"] else "MISS"
        print(f"[{idx+1:03}/{n_total}] {mark} expect={expected:4} pred={predicted:4} {elapsed:5.1f}s | {text[:48]}", flush=True)
        time.sleep(args.sleep)

    if args.resume:
        for idx, row in sorted(done.items()):
            if all(r["index"] != idx for r in rows):
                rows.append(row)
        rows.sort(key=lambda r: r["index"])

    summarize(rows, prompt_id, run, model)
    summary_path = save_summary(rows, model, prompt_id, run)
    rebuild_comparison_md()
    rebuild_index_md()
    print(f"Saved summary: {summary_path}", flush=True)
    print(f"Updated: {RESULTS_DIR / 'comparison.md'}", flush=True)
    print(f"Updated: {RESULTS_ROOT / 'INDEX.md'}", flush=True)


if __name__ == "__main__":
    main()
