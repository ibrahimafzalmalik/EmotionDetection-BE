# Deployment Guide

## Repository layout

- `EmotionDetection-BE/backend` — FastAPI + PyTorch (CPU) inference API  
- `EmotionDetection-BE/fer_project` — model code, training, and `outputs/` (checkpoints, plots)  
- `EmotionDetection-BE/Dockerfile` — builds the API image from this folder  
- `EmotionDetection-FE` — Next.js UI  

Push this repo to GitHub, then point each host at the paths below.

## Model weights and plots

`best_model.pth` and PNGs under `fer_project/outputs/` are required for a working API. Root `.gitignore` ignores `*.pth` by default. For a public demo you can either:

- Remove `*.pth` from `.gitignore` for that file only (or use [Git LFS](https://git-lfs.github.com/) for large binaries), **or**  
- Store weights in a release/download step on the host.

## Frontend (Vercel)

The UI is in the separate repo **EmotionDetection-FE** (not a subfolder of this one).

1. [vercel.com](https://vercel.com) → **Add New** → **Project** → import **EmotionDetection-FE**.  
2. **Root Directory**: leave the repo root (default).  
3. Framework: **Next.js** (auto-detected).  
4. Environment variable: `NEXT_PUBLIC_API_URL` = your public API base URL (no trailing slash), e.g. `https://your-api.example.com`  
5. Deploy.

More detail: see `EmotionDetection-FE/DEPLOYMENT.md` in that repo.

## Backend — why not Vercel?

Vercel is built for **short-lived serverless** workloads. This API loads **PyTorch**, keeps a **model in memory**, and behaves like a **normal web process**. Putting that on Vercel would mean huge bundles, cold starts, tight size/time limits, and a poor fit compared to a small container or VM. **Use Vercel for the Next.js app; run the API elsewhere.**

## Backend — Docker

Build with **`EmotionDetection-BE` as the Docker build context** (repository subfolder). Any host that runs this image can use:

- **Context / root directory**: `EmotionDetection-BE`  
- **Port**: `8000`  
- **Start**: `uvicorn backend.main:app --host 0.0.0.0 --port 8000` (same as `Dockerfile` `CMD`)

### Hugging Face Spaces — step by step

A Space is its **own** Hugging Face git repo (not the same as your GitHub repo). You copy this backend into that repo, or push from your machine. Official Docker Spaces docs: [Spaces Docker](https://huggingface.co/docs/hub/spaces-sdks-docker).

#### 1. Create the Space

1. Open [Create new Space](https://huggingface.co/new-space).
2. **Owner**: your user (or an org). **Space name**: e.g. `fer-emotion-api` (becomes part of the public URL).
3. **License**: optional. **Visibility**: **Public** is simplest for a free demo URL; private Spaces need a [paid plan](https://huggingface.co/pricing) for full privacy.
4. **SDK**: choose **Docker** (not Gradio).
5. **Hardware**: **CPU basic** (free tier is fine for this CPU-only image).
6. Click **Create Space**.

#### 2. Fix the port in `README.md` (required)

The platform routes HTTP to the port declared in the Space `README.md`. The default is **7860**; this API listens on **8000** (see `Dockerfile`). Without this, the Space builds but the app URL returns errors.

1. On the Space repo, open **Files** → `README.md` → **Edit**.
2. Ensure the YAML block at the very top includes **`app_port: 8000`** (not 7860). Example you can paste and adjust:

```yaml
---
title: FER Emotion API
emoji: 👁
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---
```

3. Save (commit).

#### 3. Put this backend’s files into the Space repo

The Space root must contain at least: `Dockerfile`, `backend/`, `fer_project/` (same layout as this GitHub repo’s backend root).

**Recommended (git on your PC):**

1. On the Space page: **⋮** (or **Settings**) → **Clone repository** → copy the HTTPS URL.
2. Locally:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME
```

3. Copy from your **EmotionDetection-BE** clone into this folder (overwrite the auto-generated `Dockerfile` if any): `Dockerfile`, entire `backend/`, and a **slim** `fer_project/` tree (inference only):

   - **Include**: `fer_project/config.py`, `fer_project/__init__.py`, `fer_project/README.md`, `fer_project/models/`, `fer_project/outputs/` (checkpoint, plots, `history.json`, etc.).
   - **Omit** `fer_project/data/` (tens of thousands of training images — not used by the API and makes `git push` very slow).

   Do **not** copy `.git` from GitHub.

4. Commit and push:

```bash
git add .
git commit -m "Add FER FastAPI Docker backend"
git push
```

**Git authentication:** Hugging Face does not accept your account password for `git push`. Create a **User Access Token** under [Settings → Access Tokens](https://huggingface.co/settings/tokens) (scope: write). Use it as the **password** when Git prompts after `git push`, or run once:

`git remote set-url origin https://USER:YOUR_TOKEN@huggingface.co/spaces/USER/SPACE_NAME.git`

**Windows + Git LFS + Hub:** LFS uploads for Spaces often use the **Xet** transfer adapter, which needs the **`git-xet`** binary on the `PATH` seen by Git’s MSYS shell. If `git push` fails with `git-xet: command not found` after “Uploading LFS objects”, install [Git Xet](https://huggingface.co/docs/hub/xet/using-xet-storage#git) and either add it to your user `PATH` or run push from **Git Bash** with `export PATH="/c/Users/YOU/AppData/Local/git-xet:$PATH"` (adjust to where `git-xet.exe` lives).

The Space will **rebuild** automatically (watch **Logs** / **Build** on the Space page). First build can take several minutes (PyTorch install).

#### 4. Your API base URL (for Vercel)

When the Space is **Running**, open the **App** tab. The browser URL is your API origin, typically:

`https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space`

Rules for the frontend env var:

- Use **`https://`**, no path, **no trailing slash**.
- Example: `https://jane-fer-emotion-api.hf.space`

Check: open `https://…hf.space/health` — you should see JSON with `"status":"ok"`.

Optional runtime env on the Space (**Settings** → **Variables and secrets**): e.g. `MODEL_DEVICE=cpu` (matches `render.yaml`; the app defaults to CPU anyway).

#### 5. Connect the frontend on Vercel

1. Deploy or open your **EmotionDetection-FE** project on [Vercel](https://vercel.com).
2. **Settings** → **Environment Variables** → add for **Production** (and **Preview** if you want):

   - **Name**: `NEXT_PUBLIC_API_URL`  
   - **Value**: the same origin as in step 4, e.g. `https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space`

3. **Save**, then **Deployments** → **⋯** on the latest deployment → **Redeploy** (env vars starting with `NEXT_PUBLIC_` are applied at **build** time).

4. Open your Vercel site and try inference; the browser calls the HF URL directly.

**Note:** Free Spaces **sleep** when idle; the first request after sleep can take a long time while the container wakes and loads the model.

### Other platforms

Many “free” PaaS products now ask for a **card for verification** (Render, Koyeb, etc.). If you want **zero** card, prioritize **Hugging Face Spaces**, a **university/organization** cluster, or a **small VPS** with a provider you already use.

`render.yaml` in this folder is optional legacy config for Render (set the service **root directory** to `EmotionDetection-BE` in the Render UI if you use it).

## After deploy

1. Copy the API HTTPS origin.  
2. Set `NEXT_PUBLIC_API_URL` on Vercel to that origin.  
3. Redeploy the frontend if you change env vars.

Cold starts on free tiers are normal; first request after idle may take tens of seconds.
