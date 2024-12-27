# Flow Edit docker

Dockerization of the [Hugging Face Flow Edit Space](https://huggingface.co/spaces/fallenshock/FlowEdit)

![Flow Edit](https://matankleiner.github.io/flowedit/resources/teaser.png)

 * Space: [https://huggingface.co/spaces/fallenshock/FlowEdit](https://huggingface.co/spaces/fallenshock/FlowEdit)

## Requisites

[Docker](https://docs.docker.com/desktop/) and [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) must be installed.

## Usage

### Download the image from the Docker Hub

You can download the image and run it

```bash
docker pull maximofn/flow_edit:latest
./run_app.sh
```

### Build the image

Or you can build the image and run it

```bash
./build_docker_image.sh
./run_app.sh
```