# Step 1: Use the official lightweight Python image
FROM python:3.11-slim

# Step 2: Set the inside working directory
WORKDIR /app

# Step 3: Copy only dependency configurations first to leverage caching
COPY requirements.txt .

# Step 4: Install your project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the remaining application source code 
COPY . .

# Step 6: Expose the networking port Streamlit listens on
EXPOSE 8501

# Step 7: Define the terminal command to launch the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]