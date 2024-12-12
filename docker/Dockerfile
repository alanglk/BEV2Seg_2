FROM pytorch/pytorch:latest

ARG USER_ID
ARG GROUP_ID

# Install general dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 sudo -y

# /lib directory allocates all the libraries developed and
#   necessary for the script's execution
WORKDIR /libraries

# Install python deppendencies and developed libraries
COPY requirements.txt /libraries/
RUN pip install --no-cache-dir -r requirements.txt
COPY ./oldatasets /libraries/oldatasets
RUN pip install --no-cache-dir -e oldatasets/
ENV PYTHONPATH=/libraries

# Copy the scripts folder
WORKDIR /
COPY ./scripts /scripts
RUN chmod +x /scripts/*.py

# # Create nonroot user
RUN groupadd -g ${GROUP_ID} user && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} user && \
    echo 'user ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/90-user && \
    passwd -d user 
USER user

# Define ENTRYPOINT para ejecutar scripts en /scripts
# ENTRYPOINT ["/bin/bash", "-c", "/scripts/$1"]

# CMD opcional para ejecutar un script específico por defecto (puede ser sobreescrito al correr el contenedor)
CMD ["/bin/bash"]

