# Use official Python base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /inFILE

# Copy the content of your local directory to the container
COPY . /inFILE

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install the required dependencies
RUN pip install -r requirements.txt

# Expose port 7860 (FastAPI default port)
EXPOSE 7860

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
