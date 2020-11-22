| Folder | Description
| :--- | :----------
| dnnlib | StyleGAN utilities.
| encoder | Encoder files to embed real images using StyleGAN Generator network.
| ffhq_dataset | Pre-processing modules.

### Starting Docker container

We use `nvcr.io/nvidia/tensorflow:19.09-py3` docker image. **Note that you need to use this specific docker image version only as the model is likely incompatible with other versions.** To run the docker container:
```
docker run --gpus all -it -p 8600:8500 -p 8601:8501 -p 5930:8888 -p 8111:8111 -p 8222:8222 --name uname-weight-changing-api -v /mnt/project_path/weight-changing-api/:/opt/weight-changing-api/ nvcr.io/nvidia/tensorflow:19.09-py3
```
**Note**: Please change `uname` in `uname-weight-changing-api` to your user account, and change `/mnt/project_path/weight-changing-api/` to your project path accordingly. The port numbers are default ports in the application. You may adjust the port numbers if they are not available to you. All instructions below are based on `/opt/weight-changing-api/` as mount point in your container. Adjust your command accordingly.

### Converting the ResNet model to a format compatible with Tensorflow serving (One-Time Process)
Tensorflow serving has it's own format when hosting models. So, you need to convert `cache/finetuned_resnet.h5` model to the format that Tensorflow serving uses. Run the command below to make the conversion. The converted model can be found at `models/`.

```
python export_saved_model.py
```

### Installing and Running Tensorflow serving (One-Time Process)

Run these commands to install Tensorflow serving.

```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list

curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -

apt-get update && apt-get install tensorflow-model-server
```
Run the command below to host the model using TF model server.
```
tensorflow_model_server --rest_api_port=8501 --model_name=resnet --model_base_path=/opt/weight-changing-api/models/ResNet/
```

### Installing the dependencies:
```
pip install -r requirements.txt
```

## Pipeline

### Step 0: Provide input data
Save all the source images under `raw_images/` folder.

### Step 1: Pre-processing
Extract faces from the source images and resize to `1024x1024` resolution.
```
python align_images.py $PATH_TO_SOURCE_IMAGES$ $PATH_TO_ALIGNED_IMAGES$.
```
__Defaults:__ $PATH_TO_SOURCE_IMAGES$=`raw_images/` and $PATH_TO_ALIGNED_IMAGES$=`aligned_images/`.
__Output:__ Aligned images at `1024x1024` resolution that are saved in the `aligned_images/` folder.

### Step 2: Latent Space Embedding
Embed the aligned images into the StyleGAN manifold.
```
CUDA_VISIBLE_DEVICES=$DEVICE_ID$ python encode_images.py $PATH_TO_ALIGNED_IMAGES$ $PATH_TO_GENERATED_IMAGES$ $PATH_TO_LATENT_CODES$
```
__Defaults:__ $PATH_TO_ALIGNED_IMAGES$=`aligned_images/`, $PATH_TO_GENERATED_IMAGES$=`generated_images/`, and `$PATH_TO_LATENT_CODES$=dlatents/`.
__Output:__ Embedded images are saved in the `generated_images/` folder and the corresponding latent codes in `dlatents/` folder.

### Step 3: Facial Weight Transformation
Generate thin to fat transformations by steering the latent codes along the `weight attribute direction`.
```
CUDA_VISIBLE_DEVICES=$DEVICE_ID$ python transform_images.py $PATH_TO_LATENT_CODES$
```
__Defaults:__ $PATH_TO_LATENT_CODES$=`dlatents/`.
__Output:__ `-5`, `-3`, `0`, `+3`, and `+5` transformations saved in the `transformed_images/` folder.

**Note**: Adjust `CUDA_VISIBLE_DEVICES` to an available GPU.


**Note**: -  This GitHub repo doesn't contain the `cache` folder where all the pre-trained models are saved. You can download the files from [here](https://drive.google.com/drive/folders/1pQJi4ql-jzmtGq48VCkIcMAaudPwNC4v?usp=sharing) and put them in a folder with the same name and place it at the current folder level.
