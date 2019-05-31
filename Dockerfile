FROM vitaflow-base
LABEL maintainer="Sampath Kumar M"

# Vitaflow Repo
WORKDIR /app
COPY . /app

# Requirements
RUN pip install -r vitaflow/annotate_server/requirements.txt

# Testing
make east_ocr_pipeline
# Cleanup - testing residues
make data_cleanup

