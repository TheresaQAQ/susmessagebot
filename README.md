# susmessagebot

An AI-powered Discord moderation bot that detects suspected scams for real-time administrator review using semantic understanding, RAG, and a Human-in-the-Loop feedback system.

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
4. Examples and system prompt are fed to SiliconFlow (`Qwen/Qwen3.5-4B`), which classifies content as `BAN`, `SAFE`, or `REVIEW` when moderation is unavailable.
5. A `BAN` classification is sent to server administrators by DM with the original message left in place. The bot does not automatically delete, strike, or ban the sender.
6. Administrators choose **Delete & Ban**, **Delete**, or **False Alarm** from the review DM. The sender is notified only after an administrator deletes the message or chooses Delete & Ban.
7. Admin feedback is used to update ChromaDB in real time and sync `susmessagebot/seeds.py` to the GitHub repository via the GitHub API — keeping the repository as the source of truth for all labelled examples. (Human-in-the-Loop)
8. Every classification, ban, and false positive is tracked as a Prometheus metric, scraped by Grafana Alloy, and visualized in a live Grafana Cloud dashboard.
9. Group and member counts are tracked automatically — every new group/server the bot is added to is recorded, with member counts updated daily.

# Tech Stack:

1. **LLM:** SiliconFlow (`Qwen/Qwen2.5-7B-Instruct`)
2. **Vector Store:** ChromaDB
3. **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
4. **Bot Framework:** discord.py
5. **Hosting:** VPS + Docker Engine + GHCR（生产机无 Compose CLI，CD 使用 `docker run`）
6. **CI/CD:** GitHub Actions (test → build/push image → SSH deploy)
7. **Example Sync:** GitHub API
8. **Observability & Monitoring:** Prometheus (`prometheus_client`) + Grafana Alloy + Grafana Cloud

## Human-in-the-Loop (HITL) Feedback System

Every time the bot flags suspicious content, admins receive three persistent review buttons by DM while the original message remains visible:

- **🚫 Delete & Ban** — deletes the original message, bans the sender, and adds the message as a `BAN` example
- **🗑️ Delete** — deletes the original message without banning the sender and adds the message as a `BAN` example
- **❌ False Alarm** — leaves the message and sender untouched and adds the message as a `SAFE` example

AI classifications do not accumulate short-window strikes and never trigger an automatic ban. Deletion and ban actions happen only after an administrator chooses them.

This means the bot gets smarter over time with every admin correction, without any manual retraining.

_Credit: This HITL feedback idea was proposed by Dr Mo Yin, a very close and treasured friend of mine. Thank you for the the friendship!_

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
  - `DASHSCOPE_API_KEY` — Alibaba Cloud Model Studio key required for image moderation and used when SiliconFlow text/URL requests fail
  - `DASHSCOPE_BASE_URL` — optional, defaults to the Beijing OpenAI-compatible endpoint
  - `DASHSCOPE_VISION_MODEL` — optional, defaults to `qwen3-vl-flash`
  - `DISCORD_BOT_TOKEN` — Discord bot token; run with `python -m susmessagebot.bot`
  - `GITHUB_TOKEN` — GitHub Personal Access Token with `Contents: Read and Write` permission
  - `GITHUB_REPO` — this repository (e.g. `TheresaQAQ/susmessagebot`)
  - `GITHUB_BRANCH` — branch to sync examples to (e.g. `main`)

## Model Used:

`Qwen/Qwen2.5-7B-Instruct` — free model on SiliconFlow (OpenAI-compatible). `Qwen/Qwen3.5-4B` is also listed but currently often times out; keep `enable_thinking=false` for Qwen3/3.5 thinking models.

Image moderation always uses Alibaba Cloud Model Studio `qwen3-vl-flash`. Text and URL moderation use SiliconFlow first, then fall back once to the same DashScope model if SiliconFlow fails or returns an unparseable verdict. If the required image backend or a fallback is unavailable, the result is `REVIEW`.

## CI/CD (Discord on VPS + Docker)

Production path for `main`:

1. GitHub Actions runs unit tests on every PR / push.
2. On push, Actions builds `ghcr.io/<owner>/susmessagebot:<git-sha>` (CPU PyTorch wheel; no CUDA) and pushes to GHCR.
3. Actions sends the runtime `compose.yaml` over SSH（仅作配置备份 / 可选 Portainer 导入）；VPS 不 clone 源码仓库。
4. VPS 用裸 `docker pull` + `docker run` 拉起精确 SHA 镜像（适配无 Compose 插件环境）。
5. Deploy waits until `http://127.0.0.1:8001/health` reports Discord readiness; on failure it rolls back to the previous image.

Application secrets (`DISCORD_BOT_TOKEN`, `SILICONFLOW_API_KEY`, `GITHUB_TOKEN`, …) stay on the VPS in `/opt/susmessagebot-secrets/.env` and are **not** stored as GitHub Actions secrets. The bot's secrets are isolated from FeedLink.

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
# Docker Engine is enough (Compose CLI not required). Then:
sudo mkdir -p /opt/susmessagebot-secrets
sudo touch /opt/susmessagebot-secrets/.env
sudo chmod 700 /opt/susmessagebot-secrets
sudo chmod 600 /opt/susmessagebot-secrets/.env

# Edit .env directly on the VPS. At minimum, set:
# DISCORD_BOT_TOKEN, SILICONFLOW_API_KEY, SILICONFLOW_MODEL,
# DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, DASHSCOPE_VISION_MODEL,
# GITHUB_TOKEN, GITHUB_REPO, and GITHUB_BRANCH.
sudo nano /opt/susmessagebot-secrets/.env

# Stop the old systemd unit so the Discord token is not used twice:
#   sudo systemctl stop susmessagebot
#   sudo systemctl disable susmessagebot

# Push to main (or manually run the workflow). Actions uploads
# compose.yaml, pulls the image, and starts the service.
```

`compose.yaml` 会同步到 `/opt/susmessagebot/compose.yaml`（文档/可选导入用）。运行时数据在独立 volume `susmessagebot_data`，不与 FeedLink 共享。备份：

```bash
sudo mkdir -p /var/backups/susmessagebot
sudo tar -czf "/var/backups/susmessagebot/data-$(date +%F).tar.gz" \
  -C /var/lib/docker/volumes/susmessagebot_data/_data .
```

### Manual rollback

```bash
IMAGE=ghcr.io/<owner>/susmessagebot:<previous-sha>
sudo docker pull "$IMAGE"
sudo docker rm -f susmessagebot
sudo docker run -d --name susmessagebot --restart unless-stopped \
  --env-file /opt/susmessagebot-secrets/.env \
  -e DATA_DIR=/app/data -e HEALTH_PORT=8001 -e METRICS_PORT=8000 \
  -v susmessagebot_data:/app/data \
  -p 127.0.0.1:8000:8000 -p 127.0.0.1:8001:8001 \
  --health-cmd='curl -fsS http://127.0.0.1:8001/health || exit 1' \
  --health-interval=30s --health-timeout=5s --health-retries=3 \
  --health-start-period=90s \
  "$IMAGE"
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

<!-- Cursor multi-file change demo: README -->
