FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

RUN pip install --no-cache-dir \
    "httpx>=0.27" \
    "pydantic>=2.0" \
    "numpy>=1.24,<2.0" \
    "huggingface_hub>=0.24" \
    "pyyaml>=6.0" \
    "jinja2>=3.1" \
    "setuptools>=65,<81" \
    "funcparserlib" \
    "intervaltree" \
    "pytrie" \
    "attrdict3" \
    "regex" \
    "typer>=0.9" \
    "rich>=12.0"

COPY pyproject.toml prompt_native.csv entrypoint.sh /home/user/app/
COPY bench/ /home/user/app/bench/
COPY swarm_lab/ /home/user/app/swarm_lab/
COPY proof_lab/ /home/user/app/proof_lab/
COPY village_lab/ /home/user/app/village_lab/
COPY conflict_lab/ /home/user/app/conflict_lab/
COPY site/ /home/user/app/site/

RUN chmod +x /home/user/app/entrypoint.sh \
    && mkdir -p /home/user/app/runtime_vendor /home/user/app/runs/edition2 /home/user/app/runs/edition3 /home/user/app/runs/edition4 \
    && chown -R user:user /home/user/app

USER user

ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    PYTHONPATH=/home/user/app:/home/user/app/runtime_vendor \
    SEBENI_DATA_DIR=/home/user/app/runtime_vendor/beni/data

EXPOSE 7860

CMD ["/home/user/app/entrypoint.sh"]
