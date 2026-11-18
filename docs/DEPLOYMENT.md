# Deployment

Host the Next.js frontend and the FastAPI backend as two HTTPS services. Phones cannot reach `localhost` on your laptop.

## Honest limits

- EfficientNet-B0 plus PyTorch is often too large for free 512 MB hosts. Prefer a VPS or a GPU/CPU machine with at least 2 GB RAM.
- A GTX 1650 with an old CUDA 11.1 driver should use **CPU** PyTorch wheels unless you upgrade the NVIDIA driver.
- Latency depends on hardware. Measure it from `/api/detect/image`; do not quote a made-up millisecond figure.
- Public inference has cost (RAM, CPU, bandwidth) and abuse risk. Keep CORS allowlisted and rate limits on.

## Backend

1. Copy `backend/.env.example` to `.env`.
2. Set `CORS_ORIGINS` to your frontend origin, for example `https://your-app.vercel.app`.
3. Install Python 3.11–3.13, create a venv, install `requirements.txt`.
4. Place `models/deepshield_efficientnet_b0.pt` and `models/model_metadata.json` on the server after training.
5. Run:

```text
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Put this behind HTTPS (Caddy, nginx, or a platform TLS terminator). Confirm:

```text
curl https://api.example.com/health
curl https://api.example.com/ready
```

`/ready` is `false` until a valid trained checkpoint is installed. That is expected.

Suggested hosts: a Linux VPS, Render, Fly.io, or Hugging Face Spaces with Docker and enough memory. Free serverless CPU timeouts may be too short for video.

## Frontend

1. Copy `frontend/.env.example` to `frontend/.env.local`.
2. Set `NEXT_PUBLIC_API_URL=https://api.example.com` (no trailing slash).
3. `npm install` then `npm run build` then `npm start`, or import the `frontend` folder into Vercel/Netlify.

Rebuild after changing `NEXT_PUBLIC_API_URL`. The value is baked in at build time.

## Mobile testing (you must do this on a real phone)

1. Open the **HTTPS** frontend URL on Android Chrome and iPhone Safari.
2. Confirm the dashboard health line is `ok`.
3. If health fails, the phone is probably still pointing at `localhost`.
4. Camera capture needs HTTPS (or localhost). Start camera, deny permission once, then allow, and confirm the error text.
5. Do not claim device testing succeeded unless those steps were run on hardware.

## CORS

The API allowlist is explicit. If the browser blocks requests, add the exact frontend origin to `CORS_ORIGINS`.
