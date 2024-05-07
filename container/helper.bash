#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ISAACGYM_DIR="$SCRIPT_DIR/isaacgym"
TBAIBINDINGS_DIR="$SCRIPT_DIR/tbai_bindings"
TBAIISAAC_DIR="$SCRIPT_DIR/../"

if [[ $1 == "--build_docker" ]]; then

  if [ ! -d $ISAACGYM_DIR ]; then
    echo "Put your isaacgym copy into this folder. It should be located at $ISAACGYM_DIR"
    exit
  fi

  if [ ! -d $TBAIBINDINGS_DIR ]; then
    echo "Put your tbai_bindings copy into this folder. It should be located at $TBAIBINDINGS_DIR"
    exit
  fi

  echo "isaacgym directory found: $ISAACGYM_DIR"
  echo "tbai_bindings directory found: $TBAIBINDINGS_DIR"

  echo "Building container..."
  DOCKERIMAGE_NAME="tbai_isaac"
  docker build -t $DOCKERIMAGE_NAME -f $SCRIPT_DIR/Dockerfile $SCRIPT_DIR
  exit
fi

if [[ $1 == "--build_singularity" ]]; then
  SINGULARITYIMAGE_NAME="tbai_isaac.sif"
  singularity build --sandbox $SINGULARITYIMAGE_NAME docker-daemon://tbai_isaac:latest
  exit
fi

if [[ $1 == "--run_docker" ]]; then
  DOCKERIMAGE_NAME="tbai_isaac"
  docker run --rm -it --privileged --runtime=nvidia $DOCKERIMAGE_NAME
  # --rm ... remove container after it exits
  # -it  ... interactive seshion
  # --runtime=nvidia ... gives access to nvidia gpus
  # --privileged     ... without this gpu is inaccessible
  exit
fi

if [[ $1 == "--run_singularity" ]]; then
  mkdir -p ./tmp
  mkdir -p ./wormhole

  SINGULARITYIMAGE_NAME="tbai_isaac.sif"
  singularity exec --nv --writable --no-home --containall \
  --bind ..:/home/tbai/tbai_isaac/ \
  --bind ./tbai_bindings:/home/tbai/tbai_bindings/src/tbai_bindings \
  --bind ./tmp:/tmp \
  --bind ./wormhole:/home/tbai/wormhole \
  $SINGULARITYIMAGE_NAME bash -c "cd /home/tbai && echo $(pwd) && rm ./setup.bash && ln -s ./tbai_isaac/container/setup.bash . && echo done && bash"
  # --nv ... gives access to nvidia gpus

  rm -rf ./tmp
  exit
fi

if [[ $1 == "--install_docker" ]]; then 
    # Update system
  sudo apt update

  # Install dependencies
  sudo apt-get install -y ca-certificates curl 
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc

  # Install docker
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  # Test installation
  sudo docker run hello-world
  exit
fi

if [[ $1 == "--install_singularity" ]]; then
  # Download singularity deb package
  NAME=apptainer_1.3.0_amd64.deb
  wget https://github.com/apptainer/apptainer/releases/download/v1.3.0/${NAME}

  # Install singularity
  sudo apt install ./${NAME}

  # Check installation
  echo "$(singularity --version)"

  # Clean up
  rm ${NAME}
  exit
fi

if [[ $1 == "--pack_singularity" ]]; then
  tar -cf tbai_isaac_sing.tar.gz $SCRIPT_DIR/.. # Tar without compression - faster
  exit
fi

if [[ $1 == "--pack_builder" ]]; then
  if [ ! -d $ISAACGYM_DIR ]; then
    echo "Put your isaacgym copy into this folder. It should be located at $ISAACGYM_DIR"
    exit
  fi

  if [ ! -d $TBAIBINDINGS_DIR ]; then
    echo "Put your tbai_bindings copy into this folder. It should be located at $TBAIBINDINGS_DIR"
    exit
  fi

  if [ -d tbai_isaac.sif ]; then
    echo "tbai_isaac.sif already exists. Remove it first."
    exit
  fi

  tar -cf tbai_isaac_builder.tar.gz ${TBAIISAAC_DIR}
  exit
fi

echo "Usage: ./helper.bash [--build_docker|--build_singularity|--run_docker|--run_singularity|--install_docker|--install_singularity|--pack_singuarity|--unpack_singularity|--pack_builder]"