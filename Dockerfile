FROM python:3.10.1-bullseye
COPY . /app
WORKDIR /app
RUN apt update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 espeak libatlas-base-dev tk
RUN pip install numpy
RUN pip install scikit-learn
RUN pip install scipy
RUN pip install --upgrade pip
RUN pip install wheel
RUN pip install Mapping
RUN pip install -r requirements.txt
RUN python -m spacy download en
RUN apt-get update && \
    apt-get clean;
EXPOSE 80
RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml
WORKDIR /app
ENTRYPOINT ["streamlit", "run"]
CMD ["Texter.py", "--server.enableCORS=false", "--server.enableWebsocketCompression=false", "--server.enableXsrfProtection=false"]