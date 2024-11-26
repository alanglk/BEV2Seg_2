FROM pytorch/pytorch:latest

# ARG USER_ID
# ARG GROUP_ID

# /lib directory allocates all the libraries developed and
#   necessary for the script's execution
WORKDIR /libraries

# Install python deppendencies and developed libraries
COPY requirements.txt /libraries/
RUN pip install --no-cache-dir -r requirements.txt
COPY ./datasets /lib/datasets
ENV PYTHONPATH="/libraries/datasets"

# Copy the scripts folder
COPY ./scripts /scripts
WORKDIR /

# # Create nonroot user
# RUN groupadd -g ${GROUP_ID} user && useradd -m -u ${USER_ID} -g ${GROUP_ID} user
# USER user

# CMD opcional para ejecutar un script específico por defecto (puede ser sobreescrito al correr el contenedor)
CMD ["python3 /scripts/generate_BEVDataset_from_NuImages.py"]

