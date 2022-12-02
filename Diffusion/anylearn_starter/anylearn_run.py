#!/usr/bin/env python
# coding: utf-8

# ### 初始化SDK与Anylearn后端引擎连接

from anylearn.config import init_sdk
from anylearn.interfaces.resource import SyncResourceUploader
init_sdk('http://anylearn.nelbds.cn', 'username', 'passward')
# ### 创建快速训练任务

import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from anylearn.applications.quickstart import quick_train

train_task, algo, dset, project = quick_train(
    algorithm_name="Sample_DDPM",
    algorithm_dir="../../Diffusion",
    
    model_id='MODE312334fb435fab6de8a2abc072ca', # checkpoints for diffusion models and classifiers
    model_hyperparam_name='pretrained_model',
    entrypoint="conda install -c conda-forge mpi4py openmpi; python -u anylearn_starter/classifier_sample_anylearn.py",
    output="outputs",
    algorithm_force_update=True,
    # resource_request=[{
    #     'DL2022-2': {
    #         'RTX-3090-shared': 1,
    #         'CPU': 4,
    #         'Memory': 4,
    #     }
    # }],
    
    resource_request=[{
        'QGRPd3da160211ecb6119ef94103bf12': {
            'A-100-unique': 1,
            'CPU': 12,
            'Memory': 128,
        }
    }],
    # mirror_name="QUICKSTART_CARTOPY_TENSORFLOW2.6_CUDA11",
    resource_uploader=SyncResourceUploader(),
)
print(train_task)


