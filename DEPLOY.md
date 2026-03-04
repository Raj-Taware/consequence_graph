# Deploying consequencegraph to Render (free tier)

## What gets deployed
A read-only visual demo of the neural-lam knowledge graph.
The graph is **pre-built locally** and committed to the repo — no indexing
happens at runtime, so the free tier has plenty of CPU headroom.

---

## Step 1 — Pre-build the graph locally

From your neural-lam root:

```bash
python consequencegraph/cli.py index ./neural_lam --preset neural_lam
```

This writes `.consequencegraph/cache.json`. Now copy it into the consequencegraph repo:

```bash
mkdir -p consequencegraph/neural_lam_cached
cp .consequencegraph/cache.json consequencegraph/neural_lam_cached/cache.json
```

The server in production mode loads from this cache on startup.
No network calls, no indexing, instant boot.

---

## Step 2 — Update server.py boot path

In `server.py`, the `main()` function defaults `--path` to `./neural_lam`.
For deployment, point it at the cached directory:

```python
parser.add_argument("--path", default="./neural_lam_cached")
```

Or just set it in `render.yaml` startCommand (already done).

---

## Step 3 — Push to GitHub

```bash
git add consequencegraph/
git commit -m "add consequencegraph with pre-built neural-lam graph"
git push
```

Make sure `neural_lam_cached/cache.json` is NOT in `.gitignore`.
It's ~2MB, fine to commit.

---

## Step 4 — Deploy on Render

1. Go to https://render.com and sign up (free, no credit card)
2. New → Web Service → Connect your GitHub repo
3. Render will detect `render.yaml` automatically
4. Deploy

Your URL will be: `https://consequencegraph.onrender.com` (or similar)

---

## Production behaviour

- `/api/reindex` is **disabled** (returns 403)
- Rate limit: 60 requests/minute per IP
- CORS open (anyone can embed the API)
- Free tier sleeps after 15min inactivity — first load after sleep takes ~30s

To avoid sleep on the free tier, use UptimeRobot (free) to ping `/api/stats`
every 14 minutes.

---

## Local development

```bash
# Full mode — indexes live codebase, reindex enabled
python consequencegraph/server.py --path ./neural_lam --preset neural_lam --reindex

# Production simulation locally
CONSEQUENCEGRAPH_ENV=production python consequencegraph/server.py --path ./neural_lam_cached
```
