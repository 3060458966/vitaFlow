FROM vitaflow-base
LABEL maintainer="Sampath Kumar M"

# Annotation Tool Requirement
RUN apt-get install tesseract-ocr -y
RUN pip install -r vitaflow/annotate_server/requirements.txt

CMD "sleep 1000"
# docker run -i -t  vitaflow:0.1123 -- bash
