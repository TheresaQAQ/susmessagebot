# susmessagebot

An AI-powered Discord moderation bot that detects and bans scammers in real time using semantic understanding, RAG, and a Human-in-the-Loop feedback system.

## Branding

This is proudly a @commonertech product.

## Why

Singaporeans lost a record S$1.1 billion to scams in 2024 and S$913.1 million in 2025 (Singapore Police Force). Community platforms are frequently exploited by scammers, while existing moderation tools rely on static keyword rules that are easily bypassed with character substitution and deliberate typos. This bot was built to fight back using semantic understanding instead of keyword matching.

## Architecture

![Architecture Diagram](assets/architecture.png)

## Project Structure

```text
susmessagebot/  Runtime package, prompts, and seed examples
scripts/        Evaluation and bakeoff tools
tests/          Regression tests
.github/        CI/CD workflows and GitHub templates
```

# Technical Implementation:

1. `susmessagebot/bot.py` scans incoming Discord messages and image attachments.
2. Incoming text is sent to `susmessagebot/moderator.py`, where it is first converted into an embedding (defined in `susmessagebot/vector_store.py`).
3. Input text embedding is then compared with current existing labelled examples to retrieve most similar examples. (Retrieval-Augmented Generation)
4. Examples and system prompt are in tandem fed to SiliconFlow (`Qwen/Qwen3.5-4B`), which will return a singular classification: `BAN` or `SAFE`.
5. The bot acts on the classification — deleting the message and banning the user if `BAN`, doing nothing if `SAFE`.
6. On every ban, admins are notified with two buttons: **✅ Correct Ban** or **❌ Wrong Ban**.
7. Admin feedback is used to update ChromaDB in real time and sync `susmessagebot/seeds.py` to the GitHub repository via the GitHub API — keeping the repository as the source of truth for all labelled examples. (Human-in-the-Loop)
8. Every classification, ban, and false positive is tracked as a Prometheus metric, scraped by Grafana Alloy, and visualized in a live Grafana Cloud dashboard.
9. Admins (or users pending admin approval) can report missed scams — adding them to ChromaDB, syncing to GitHub, and tracking as false negatives in the monitoring dashboard.
10. Group and member counts are tracked automatically — every new group/server the bot is added to is recorded, with member counts updated daily.

# Tech Stack:

1. **LLM:** SiliconFlow (`Qwen/Qwen2.5-7B-Instruct`)
2. **Vector Store:** ChromaDB
3. **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
4. **Bot Framework:** discord.py
5. **Hosting:** VPS (e.g. Google Cloud e2-micro) + Docker Compose + GHCR
6. **CI/CD:** GitHub Actions (test → build/push image → SSH deploy)
7. **Example Sync:** GitHub API
8. **Observability & Monitoring:** Prometheus (`prometheus_client`) + Grafana Alloy + Grafana Cloud

## Human-in-the-Loop (HITL) Feedback System

Every time the bot removes suspicious content, admins receive two review buttons by DM:

- **✅ Correct Ban** — confirms the ban and adds the message as a `BAN` example to ChromaDB and `susmessagebot/seeds.py`
- **❌ Wrong Ban** — marks it as a false positive, unbans the user, and adds the message as a `SAFE` example to ChromaDB and `susmessagebot/seeds.py`

This means the bot gets smarter over time with every admin correction, without any manual retraining.

_Credit: This HITL feedback idea was proposed by Dr Mo Yin, a very close and treasured friend of mine. Thank you for the the friendship!_

## Reporting missed scams

If the bot misses a scam (false negative), it can be manually reported:

- Use the **Report to SusMessageBot** message context menu, or mention the bot while replying to the suspicious message.
- **Admins** can immediately remove the message, ban the sender, and add the content to training examples.
- **Non-admins** submit the message for administrator review through **✅ Confirm Ban** and **❌ Dismiss** buttons.

False negatives are tracked separately in the monitoring dashboard.

## Live Monitoring Dashboard

[susmessagebot.commonertech.dev/dashboard](https://susmessagebot.commonertech.dev/dashboard)

## Additional Details:

As of date of creation (30 March 2026), I have yet to provision an Oracle Cloud VPS Instance due to consistently meeting "Host Out of Capacity" errors. This is expected as it's understandable that everyone would want to sign up for their generous free tier.
But sometimes, a brick wall is not the end of the road - you just have to find a path around it.

## Pros:

1. Lightweight, able to be run on a smaller free VPS instance (such as Google Cloud e2-micro instance)
2. No GPU/CPU overhead - faster inference than self-hosted on CPU
3. Easier setup - no Ollama installation required

## Cons:

1. Not fully self-hosted - messages are sent to SiliconFlow (potential privacy consideration)
2. Dependency on SiliconFlow free-model availability and rate limits
3. Very high-volume multi-group deployments still need monitoring

## Key Caveat:

As this bot is in the initial deployment stages, please do expect a fair number of false positives. As more people use the bot and admins participate in the HITL review, the accuracy of the bot will increase over time.
I seek your kind understanding for any teething issues.

## Setup Differences from Main Branch:

- Replace `OLLAMA_MODEL` and `OLLAMA_HOST` in config with `SILICONFLOW_API_KEY` (and optional `SILICONFLOW_MODEL`)
- No `ollama pull` step required
- Add the following to your `.env` file:
  - `SILICONFLOW_API_KEY` — obtain from [cloud.siliconflow.cn](https://cloud.siliconflow.cn)
  - `SILICONFLOW_MODEL` — optional, defaults to `Qwen/Qwen2.5-7B-Instruct`
  - `DISCORD_BOT_TOKEN` — Discord bot token; run with `python -m susmessagebot.bot`
  - `GITHUB_TOKEN` — GitHub Personal Access Token with `Contents: Read and Write` permission
  - `GITHUB_REPO` — your forked repository (e.g. `yourusername/susmessagebot`)
  - `GITHUB_BRANCH` — branch to sync examples to (e.g. `groq-approach`)

## Model Used:

`Qwen/Qwen2.5-7B-Instruct` — free model on SiliconFlow (OpenAI-compatible). `Qwen/Qwen3.5-4B` is also listed but currently often times out; keep `enable_thinking=false` for Qwen3/3.5 thinking models.

## CI/CD (Discord on VPS + Docker)

Production path for `groq-approach`:

1. GitHub Actions runs unit tests on every PR / push.
2. On push, Actions builds `ghcr.io/<owner>/susmessagebot:<git-sha>` (CPU PyTorch wheel; no CUDA) and pushes to GHCR.
3. Actions SSHs into the VPS, pulls that exact image, and restarts Docker Compose.
4. Deploy waits until `http://127.0.0.1:8001/health` reports Discord readiness; on failure it rolls back to the previous image.

Application secrets (`DISCORD_BOT_TOKEN`, `SILICONFLOW_API_KEY`, `GITHUB_TOKEN`, …) stay on the VPS in `.env` and are **not** stored as GitHub Actions secrets.

### GitHub Actions secrets

| Secret | Purpose |
|--------|---------|
| `VPS_HOST` | VPS hostname / IP |
| `VPS_USER` | SSH username |
| `VPS_SSH_KEY` | Private key for SSH deploy |
| `GHCR_USERNAME` | GHCR pull username (usually your GitHub username) |
| `GHCR_READ_TOKEN` | PAT or fine-grained token with `read:packages` (and SSO authorized if needed) |

### VPS first-time setup

```bash
# Install Docker Engine + Compose plugin, then:
git clone git@github.com:<you>/susmessagebot.git ~/susmessagebot
cd ~/susmessagebot
git checkout groq-approach

cp .env.example .env
# edit .env — set DISCORD_BOT_TOKEN, SILICONFLOW_*, GITHUB_*

mkdir -p data
# Migrate legacy host paths if you previously ran outside Docker:
#   [ -f stats.db ] && mv stats.db data/stats.db
#   [ -d chroma_db ] && mv chroma_db data/chroma_db

# Stop the old systemd unit so the Discord token is not used twice:
#   sudo systemctl stop susmessagebot
#   sudo systemctl disable susmessagebot

echo "$GHCR_READ_TOKEN" | docker login ghcr.io -u "$GHCR_USERNAME" --password-stdin
export IMAGE=ghcr.io/<owner>/susmessagebot:latest
docker compose pull
docker compose up -d
curl -fsS http://127.0.0.1:8001/health
```

Persistent runtime data lives in `~/susmessagebot/data/` (`stats.db`, `chroma_db/`). Back it up before risky changes:

```bash
tar -czf "susmessagebot-data-$(date +%F).tar.gz" -C ~/susmessagebot data
```

### Manual rollback

```bash
cd ~/susmessagebot
export IMAGE=ghcr.io/<owner>/susmessagebot:<previous-sha>
docker compose pull
docker compose up -d
```

### Local Discord run (without Docker)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-vps.txt
cp .env.example .env   # fill tokens
python -m susmessagebot.bot
```

Health: `http://127.0.0.1:8001/health` (200 only after Discord gateway ready)  
Metrics: `http://127.0.0.1:8000/metrics`

## Sponsorship

Running this bot at scale requires paid infrastructure. If this project has been useful to you and you'd like to help cover hosting costs or support further development, consider sponsoring:

- ⭐ Star the repo to show support
- ❤️ [GitHub Sponsors](https://github.com/sponsors/0mgABear)
- 📧 Contact: hello@commonertech.dev
- ☕ [Ko-fi](https://ko-fi.com/commonertech)

Every contribution helps keep the bot running and the project maintained. Thank you! 🙏
