# Group5

### Run Docker

```
docker build .
```

```
docker run --gpus all <IMAGE_ID> [arguments]
```

In case GPUs are not recognized by the docker container, make sure `nvidia-container-toolkit` is installed and the docker daemon is restarted.

### Run Manually

#### Dependency installation
```
pip install -r requirements.txt
```

### Arguments

`--do_train <True/False>` ...
`--do_eval <True/False>` ...
