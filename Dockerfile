FROM python:3.7-slim

WORKDIR /app

# Copy the code
COPY . /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 espeak libatlas-base-dev tk

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -m spacy download en

# Expose the port
EXPOSE 80

# Set up Streamlit configuration
RUN mkdir -p /root/.streamlit
RUN cp /app/config.toml /root/.streamlit/config.toml
RUN cp /app/credentials.toml /root/.streamlit/credentials.toml

# Set the working directory
WORKDIR /app

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run"]
CMD ["Texter.py", "--server.enableCORS=false", "--server.enableWebsocketCompression=false", "--server.enableXsrfProtection=false"]
