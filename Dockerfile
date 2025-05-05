FROM python:3.10

WORKDIR /inFILE

COPY . /inFILE

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
