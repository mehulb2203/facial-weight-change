## Back-End

| Folder | Description
| :--- | :----------
| 01-Pre_processing | Step-1 of the pipeline.
| 02-Latent_Space_Embedding | Step-2 of the pipeline.
| 03-Facial_Weight_Transformation | Step-3 of the pipeline.
| dnnlib | StyleGAN utilities.
| encoder | Encoder files to embed real images using StyleGAN Generator network.

### Starting Docker container

We use `nvcr.io/nvidia/tensorflow:19.09-py3` docker image. **Note that you need to use the specific docker image version because, the application serializes model training. The model is likely incompatible with other versions.** To run the docker container:
```
docker run --gpus all -it -p 8600:8500 -p 8601:8501 -p 5930:8888 -p 8111:8111 -p 8222:8222 --name uname-weight-changing-api -v /mnt/project_path/weight-changing-api/:/opt/weight-changing-api/ nvcr.io/nvidia/tensorflow:19.09-py3
```
**Note**: Please change `uname` in `uname-weight-changing-api` to your user account, and change your `/mnt/project_path/weight-changing-api/` project path accordingly. The port numbers are default ports in the application. You may adjust the port numbers if they are not available to you. All instructions below are based on `/opt/weight-changing-api/` as mount point in your container. Adjust your command accordingly.

### Preparing our model for inference using Tensorflow serving (One-Time Process)
Tensorflow serving has it's own format of hosting the models that makes it efficient during inference. So, you need to convert your `myModel.h5` or `myModel.hdf5` models into the format that Tensorflow serving uses. Also, change the `model_path` and `export_path` in `export_saved_model.py` to the location of your pre-trained model (which is in `.h5` or `.hdf5` format) and the location where you want to save the converted model, respectively.

```
python export_saved_model.py
```

### Installing and Running Tensorflow serving

Tensorflow serving hosts our pre-trained `ResNet` image classifier model. You can get the model from [here](https://drive.google.com/file/d/1Jrz5qw8Hnp0C0C2iiYx7w13l5ciHUKH0/view?usp=sharing). After the docker container is ready, run these commands to install Tensorflow serving.

```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list

curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -

apt-get update && apt-get install tensorflow-model-server

tensorflow_model_server --rest_api_port=8501 --model_name=resnet --model_base_path=/opt/weight-changing-api/models/ResNet/
```

### Install all the dependencies:
```
pip install -r requirements.txt
```


**Note**: -  This GitHub repo doesn't contain the `cache` folder where all the models we use, exist. You can download the files from [here](https://drive.google.com/drive/folders/1pQJi4ql-jzmtGq48VCkIcMAaudPwNC4v?usp=sharing) and put them in a folder named `cache` and place the folder at the current level.
