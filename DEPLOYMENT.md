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

1. [vercel.com](https://vercel.com) → **Add New** → **Project** → import this GitHub repo.  
2. **Root Directory**: `EmotionDetection-FE`  
3. Framework: **Next.js** (auto-detected).  
4. Environment variable: `NEXT_PUBLIC_API_URL` = your public API base URL (no trailing slash), e.g. `https://your-api.example.com`  
5. Deploy.

## Backend — why not Vercel?

Vercel is built for **short-lived serverless** workloads. This API loads **PyTorch**, keeps a **model in memory**, and behaves like a **normal web process**. Putting that on Vercel would mean huge bundles, cold starts, tight size/time limits, and a poor fit compared to a small container or VM. **Use Vercel for the Next.js app; run the API elsewhere.**

## Backend — Docker

Build with **`EmotionDetection-BE` as the Docker build context** (repository subfolder). Any host that runs this image can use:

- **Context / root directory**: `EmotionDetection-BE`  
- **Port**: `8000`  
- **Start**: `uvicorn backend.main:app --host 0.0.0.0 --port 8000` (same as `Dockerfile` `CMD`)

### Hugging Face Spaces (often the easiest free ML demo)

Spaces can run a **Docker** Space from your repo. Set the Space’s base directory to `EmotionDetection-BE` if the platform supports it, or copy this folder’s `Dockerfile` into the Space. See [Spaces Docker documentation](https://huggingface.co/docs/hub/spaces-sdks-docker).

### Other platforms

Many “free” PaaS products now ask for a **card for verification** (Render, Koyeb, etc.). If you want **zero** card, prioritize **Hugging Face Spaces**, a **university/organization** cluster, or a **small VPS** with a provider you already use.

`render.yaml` in this folder is optional legacy config for Render (set the service **root directory** to `EmotionDetection-BE` in the Render UI if you use it).

## After deploy

1. Copy the API HTTPS origin.  
2. Set `NEXT_PUBLIC_API_URL` on Vercel to that origin.  
3. Redeploy the frontend if you change env vars.

Cold starts on free tiers are normal; first request after idle may take tens of seconds.
