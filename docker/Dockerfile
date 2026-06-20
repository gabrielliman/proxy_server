# Use the specified base image
FROM willianjunior/vllm-0.19.0

# Set a working directory inside the container (optional but recommended)
WORKDIR /app

# Copy the requirements file from your local machine into the container
COPY requirements.txt .

# Install the packages listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN echo "teste"