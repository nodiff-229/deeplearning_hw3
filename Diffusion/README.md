# Hints for runing this code


## Environment 
Run the following scripts in the root path of ```Diffusion```.

```
pip install -r requirements.txt
pip install mpi4py ;
```
If your system lack dependencies for ```mpi4py```, you can try to install it with ```conda```:

```
conda install -c conda-forge mpi4py openmpi
```

## Run
Since training and sampling from diffusion models is rather time-consuming, 
we provide you with open-sourced pretrained diffusion checkpoints, and your tasks 
are focus on the implementation of diffusion sampling process and case studies.
To run this code, you need to modify classifier_sample.sh.

```bash
export CUDA_VISIBLE_DEVICES=0
MODEL_FLAGS="--attention_resolutions 32,16,8 
            --class_cond True
            --rescale_timesteps True
            --diffusion_steps 1000 
            --dropout 0.1
            --image_size 64
            --learn_sigma True
            --noise_schedule cosine
            --num_channels 192
            --num_head_channels 64
            --num_res_blocks 3
            --resblock_updown True
            --use_new_attention_order True
            --use_fp16 True
            --use_scale_shift_norm True"
python classifier_sample.py $MODEL_FLAGS \
      --classifier_scale 1.0 \ 
      --classifier_path path/to/classifier.pt \
      --classifier_depth 4 \
      --model_path .path/to/model.pt \
      --save_dir outputs $SAMPLE_FLAGS \
      --class_index -1 \
      --use_ddim False
```

The following parameters are concerned in this task.

```--diffusion_steps``` int number in the interval [0,1000];

```--image_size``` depend on the checkpoint you use, we recommend 64 for the time and the computation consumption.

```--classifier_path``` path/to/classifier.pt

```--model_path``` .path/to/model.pt

```--class_index ``` -1 for free generation, int number in [0, 999] for class conditioned generation\

```--use_ddim ``` False for DDPM True for DDIM.
## Checkpoints download
You can download pretrained checkpoints from Tsinghua cloud mirror:
https://cloud.tsinghua.edu.cn/d/d596b9ae5a9a49788af5/

or you can directly visit the original project:
https://github.com/openai/guided-diffusion

## DDIM Implementation
Although DDIM has theoratical difference with DDPM, the implementations of sampling processes are alike.
We mask the sample function in the original implementation, and you need to understand the sampling processes 
implemented in ```guided_diffusion/gaussian_diffusion``` and fill the blank in the ```ddim_sample``` function.

Additional:
- In your implementation do not need to consider gradient-based conditioning induced by the classifier.
- You are not allowed to directly copy code from existing projects or use packages like ```diffuser```.

## Anylearn 
We provide start code for anylearn in ```anylearn_starter```. You can run the followings with your username and passward.
```
cd anylearn_starter
python anylearn_run.py
```