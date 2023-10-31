# Use the official Python image as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /Users/charlenehack/Desktop/Sam/fastapi_docker_ml
# List the contents of the working directory
RUN ls -la



# Copy your FastAPI app and model files into the container
COPY app.py .
COPY zipcode_demographics.csv .
COPY model_features.json .
COPY model.pkl .

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install FastAPI, Uvicorn, and other dependencies
RUN pip install fastapi uvicorn pandas scikit-learn python-multipart

# Expose the port the app will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


#build the image and run the container on the specified port, naviage to the swagger docs to test the api
#docker build --no-cache -t fastapi-app .

#docker run -d -p 8000:8000 fastapi-app

