import { mkdir, readFile, writeFile, appendFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { experimental_evaluate as evaluate } from "ai";
import { POLICY_CRITERIA, POLICY_INSTRUCTIONS } from "./policy.ts";

type GoldRow = {
  index: number;
  text: string;
  expected: "BAN" | "SAFE";
  tag: string;
};

type ResultRow = GoldRow & {
  predicted: "BAN" | "SAFE";
  ok: boolean;
  probability: number | null;
  seconds: number;
  inputTokens?: number;
};

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const GOLD_PATH = join(ROOT, "data", "gold.json");
const OUT_DIR = join(ROOT, "eval_results", "typesafe-ai_jev");
const RESULTS_PATH = join(OUT_DIR, "results.jsonl");
const SUMMARY_PATH = join(OUT_DIR, "summary.json");

const QUESTIONS = {
  verdict: {
    type: "choice" as const,
    instructions: POLICY_INSTRUCTIONS,
    criteria: POLICY_CRITERIA,
  },
};

function parseArgs(argv: string[]) {
  const get = (name: string, fallback: string) => {
    const i = argv.indexOf(name);
    return i >= 0 ? argv[i + 1] : fallback;
  };
  return {
    limit: Number(get("--limit", "0")),
    concurrency: Number(get("--concurrency", "1")),
    sleepMs: Number(get("--sleep-ms", "2500")),
    resume: argv.includes("--resume"),
  };
}

async function loadGold(): Promise<GoldRow[]> {
  return JSON.parse(await readFile(GOLD_PATH, "utf8")) as GoldRow[];
}

async function loadDone(): Promise<Map<number, ResultRow>> {
  try {
    const text = await readFile(RESULTS_PATH, "utf8");
    const map = new Map<number, ResultRow>();
    for (const line of text.split("\n")) {
      if (!line.trim()) continue;
      const row = JSON.parse(line) as ResultRow;
      map.set(row.index, row);
    }
    return map;
  } catch {
    return new Map();
  }
}

let writeChain = Promise.resolve();

function appendResult(row: ResultRow) {
  writeChain = writeChain.then(() =>
    appendFile(RESULTS_PATH, `${JSON.stringify(row)}\n`, "utf8"),
  );
  return writeChain;
}

function explainGatewayError(error: unknown): never {
  const message = error instanceof Error ? error.message : String(error);
  if (message.includes("credit card") || message.includes("customer_verification_required")) {
    throw new Error(
      "Vercel AI Gateway 需要账号绑定信用卡后才能调用 Jev。打开 https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai%3Fmodal%3Dadd-credit-card 添加卡片，然后重新运行 npm run eval:ads",
    );
  }
  if (message.includes("Zero Data Retention") || message.includes("ZDR")) {
    throw new Error(
      "当前是 Hobby 套餐，不能开 Zero Data Retention。已从评测请求里去掉该选项后再跑。",
    );
  }
  throw error;
}

function isRateLimit(error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    message.includes("rate_limit") ||
    message.includes("RateLimit") ||
    message.includes("high demand") ||
    message.includes("maxRetriesExceeded")
  );
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function classify(text: string) {
  const started = performance.now();
  for (let attempt = 1; attempt <= 8; attempt += 1) {
    try {
      const result = await evaluate({
        model: "typesafe-ai/jev",
        state: { message: text },
        questions: QUESTIONS,
        maxRetries: 0,
      });
      const answer = result.answers.verdict;
      return {
        predicted: answer.choice,
        probability: answer.probabilities?.[answer.choice] ?? null,
        inputTokens: result.usage.inputTokens,
        seconds: (performance.now() - started) / 1000,
      };
    } catch (error) {
      if (isRateLimit(error) && attempt < 8) {
        console.log(`rate limited, wait 65s then retry (${attempt}/8)`);
        await sleep(65_000);
        continue;
      }
      explainGatewayError(error);
    }
  }
  throw new Error("classify exhausted retries");
}

async function mapPool<T, R>(
  items: T[],
  concurrency: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  async function worker() {
    while (true) {
      const i = next;
      next += 1;
      if (i >= items.length) return;
      results[i] = await fn(items[i], i);
    }
  }
  await Promise.all(
    Array.from({ length: Math.max(1, concurrency) }, () => worker()),
  );
  return results;
}

function summarize(rows: ResultRow[]) {
  const total = rows.length;
  const correct = rows.filter((r) => r.ok).length;
  const tp = rows.filter((r) => r.expected === "BAN" && r.predicted === "BAN").length;
  const tn = rows.filter((r) => r.expected === "SAFE" && r.predicted === "SAFE").length;
  const fp = rows.filter((r) => r.expected === "SAFE" && r.predicted === "BAN").length;
  const fn = rows.filter((r) => r.expected === "BAN" && r.predicted === "SAFE").length;
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  const avgSeconds = total
    ? rows.reduce((sum, r) => sum + r.seconds, 0) / total
    : 0;
  const inputTokens = rows.reduce((sum, r) => sum + (r.inputTokens ?? 0), 0);

  const byPrefix = new Map<string, { ok: number; n: number }>();
  for (const row of rows) {
    const prefix = row.tag.split("-")[0] ?? row.tag;
    const bucket = byPrefix.get(prefix) ?? { ok: 0, n: 0 };
    bucket.n += 1;
    if (row.ok) bucket.ok += 1;
    byPrefix.set(prefix, bucket);
  }

  return {
    model: "typesafe-ai/jev",
    n: total,
    correct,
    accuracy: total ? correct / total : 0,
    tp,
    tn,
    fp,
    fn,
    banPrecision: precision,
    banRecall: recall,
    banF1: f1,
    avgSeconds,
    inputTokens,
    byPrefix: Object.fromEntries(byPrefix),
    mistakes: rows
      .filter((r) => !r.ok)
      .map((r) => ({
        index: r.index,
        expected: r.expected,
        predicted: r.predicted,
        tag: r.tag,
        text: r.text,
        probability: r.probability,
      })),
  };
}

function printSummary(summary: ReturnType<typeof summarize>) {
  console.log("\n========== JEV EVAL ==========");
  console.log(`Model             : ${summary.model}`);
  console.log(
    `Accuracy          : ${summary.correct}/${summary.n} = ${pct(summary.accuracy)}`,
  );
  console.log(
    `Confusion         : TP=${summary.tp} TN=${summary.tn} FP=${summary.fp} FN=${summary.fn}`,
  );
  console.log(`BAN precision     : ${pct(summary.banPrecision)}`);
  console.log(`BAN recall        : ${pct(summary.banRecall)}`);
  console.log(`BAN F1            : ${pct(summary.banF1)}`);
  console.log(`Avg seconds       : ${summary.avgSeconds.toFixed(2)}s`);
  console.log(`Input tokens      : ${summary.inputTokens}`);
  console.log("\nBy language/prefix:");
  for (const [prefix, bucket] of Object.entries(summary.byPrefix).sort()) {
    console.log(
      `  ${prefix.padEnd(6)} ${bucket.ok}/${bucket.n} = ${pct(bucket.ok / bucket.n)}`,
    );
  }
  if (summary.mistakes.length) {
    console.log(`\nMistakes (${summary.mistakes.length}):`);
    for (const row of summary.mistakes.slice(0, 30)) {
      console.log(
        `  [${String(row.index).padStart(3, "0")}] expect=${row.expected} pred=${row.predicted} tag=${row.tag} | ${row.text.slice(0, 60)}`,
      );
    }
  }
  console.log("==============================\n");
}

function pct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const gold = await loadGold();
  const done = args.resume ? await loadDone() : new Map<number, ResultRow>();
  const pending = gold.filter((row) => !done.has(row.index));
  const todo = args.limit > 0 ? pending.slice(0, args.limit) : pending;

  await mkdir(OUT_DIR, { recursive: true });
  if (!args.resume) {
    await writeFile(RESULTS_PATH, "", "utf8");
  }

  console.log(
    `Gold ${gold.length} | resume ${done.size} | this run ${todo.length} | concurrency ${args.concurrency} | sleep ${args.sleepMs}ms`,
  );

  await mapPool(todo, args.concurrency, async (item) => {
    process.stdout.write(
      `[${String(item.index + 1).padStart(3, "0")}/${gold.length}] classifying... | ${item.text.slice(0, 48)}\n`,
    );
    const result = await classify(item.text);
    const row: ResultRow = {
      ...item,
      predicted: result.predicted,
      ok: result.predicted === item.expected,
      probability: result.probability,
      seconds: Number(result.seconds.toFixed(2)),
      inputTokens: result.inputTokens,
    };
    done.set(item.index, row);
    await appendResult(row);
    console.log(
      `[${String(item.index + 1).padStart(3, "0")}/${gold.length}] ${row.ok ? "OK" : "MISS"} expect=${item.expected} pred=${row.predicted} ${row.seconds.toFixed(2)}s`,
    );
    if (args.sleepMs > 0) {
      await sleep(args.sleepMs);
    }
    return row;
  });

  const rows = gold
    .map((item) => done.get(item.index))
    .filter((row): row is ResultRow => Boolean(row));
  const summary = summarize(rows);
  await writeFile(SUMMARY_PATH, JSON.stringify(summary, null, 2), "utf8");
  printSummary(summary);
  console.log(`Saved: ${RESULTS_PATH}`);
  console.log(`Saved: ${SUMMARY_PATH}`);
}

await main();
