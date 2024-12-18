
# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY app.py .
COPY models/ ./models/
COPY templates/ ./templates/
COPY requirement.txt .


# Install any needed dependencies
RUN pip install  -r requirement.txt

# Expose the port the app runs on
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]


