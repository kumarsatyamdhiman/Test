FROM python:3.7
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install tf_slim
EXPOSE 8080
COPY . /app
CMD streamlit run --server.port 8080 --server.enableCORS false app.py