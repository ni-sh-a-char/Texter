# Use the official Python 3.11.3 image from Bullseye
FROM python:3.11.3-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the code from the current directory to the container's working directory
COPY . /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 espeak libatlas-base-dev tk

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -m spacy download en

# Expose port 80
EXPOSE 80

# Set up Streamlit configuration
RUN mkdir -p /root/.streamlit
RUN cp /app/config.toml /root/.streamlit/config.toml
RUN cp /app/credentials.toml /root/.streamlit/credentials.toml

# Set the working directory again (optional but recommended)
WORKDIR /app

# Run the Streamlit app as the entry point
ENTRYPOINT ["streamlit", "run"]

# Specify the default command for the container
CMD ["Texter.py", "--server.enableCORS=false", "--server.enableWebsocketCompression=false", "--server.enableXsrfProtection=false"]
