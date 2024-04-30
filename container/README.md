## Container deployment

To be able to train agents inside of a container, one first needs to be generated.
You can use the prepared utility script `helper.bash` that makes the container
generation and deployment a seamless process.


```bash

# Generate a docker image
bash helper.bash --generate_docker

# Generate a singularity image
bash helper.bash --generate_docker && bash helper.bash --generate_singularity

# Deploy docker container
bash helper.bash --start_docker

# Deploy singularity container
bash helper.bash --start_singularity

```