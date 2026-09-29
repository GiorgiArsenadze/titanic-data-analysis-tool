FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    SEABORN_DATA=/app/seaborn-data \
    OUTPUT_PATH=/app/output/titanic_ds_dashboard.html

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache the Titanic dataset at build time so the container runs offline
RUN python -c "import seaborn as sns; sns.load_dataset('titanic')"

COPY ds_analysis.py .

CMD ["python", "ds_analysis.py"]
