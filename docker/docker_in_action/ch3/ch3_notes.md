# How to identify a docker image?
[REGISTRYHOST:PORT/][USERNAME/]NAME[:TAG]

# Some commands.
docker pull quay.io/dockerinaction/ch3_hello_registry:latest
docker rmi quay.io/dockerinaction/ch3_hello_registry:latest
docker save -o myfile.tar busybox:latest
- If `-o` flag is omitted, the resulting file will be streamed to the terminal.
docker load -i myfile.tar
docker images -a

## using dockerfile
git clone https://github.com/dockerinaction/ch3_dockerfile.git
docker build -t dia_ch3/dockerfile:latest ch3_dockerfile

# Images and layers.
An image is actually a collection of image layers.
A layer is set of files and file metadata that is packaged and distributed as an atomic unit.

## trying pulling similar images
docker pull dockerinaction/ch3_myapp
docker pull dockerinaction/ch3_myotherapp

# UFS(union filesystem)

