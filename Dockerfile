FROM python:3.11-slim

WORKDIR /app

COPY backend/ ./backend/
COPY fer_project/ ./fer_project/

# Fail the image build if the checkpoint is missing or still a Git LFS pointer stub.
RUN python -c "from pathlib import Path; p=Path('fer_project/outputs/checkpoints/best_model.pth'); s=p.stat().st_size if p.is_file() else 0; assert s>1_000_000, f'Invalid checkpoint ({s} bytes). Push best_model.pth with Git LFS.'; print('checkpoint ok:', s)"

RUN pip install --no-cache-dir -r backend/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
