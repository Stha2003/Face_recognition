FROM python:3.10-slim

WORKDIR /

# copy code 
COPY . /

# Install OpenCV dependencies
RUN pip install opencv-python


# Install OpenCV and GUI support libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install -r requirements.txt

CMD ["python", "main.py"]