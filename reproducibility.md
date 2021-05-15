# Reproducibility

This document shows the workloads in the LambdaML paper [1] and describes how to run these workloads.

## S3

### Linear models

- Trigger function: 
[linear_s3_trigger.py](examples/lambda/s3/linear_s3_trigger.py). 
    - Set *data_bucket* and *file* according to the specific dataset.
    - Set *dataset_type* to dense_libsvm for dense dataset or sparse_libsvm for sparse dataset.
    - Set *model* to lr/svm for dense model or sparse_lr/sparse_svm for sparse model.
    - Set *optim* to grad_avg for gradient average (SGD), model_avg for model average, or admm for ADMM.
    - Set *sync_mode* to reduce, reduce_scatter or async.
    - Set other configurations and hyperparameters.
- Execution function. Choose the execution function according to sparsity and optimization algorithm.
    - Dense model, SGD or Model Average. [linear_s3_ga_ma.py](examples/lambda/s3/linear_s3_ga_ma.py)
    - Dense model, ADMM. [linear_s3_admm.py](examples/lambda/s3/linear_s3_admm.py)
    - Sparse model, SGD or Model Average. [sparse_linear_s3_ga_ma.py](examples/lambda/s3/sparse_linear_s3_ga_ma.py)
    - Sparse model, ADMM. [sparse_linear_s3_admm.py](examples/lambda/s3/sparse_linear_s3_admm.py)

    
### KMeans

### CNN models




## Elasticache


## DynamoDB


## Hybrid


## Reference
Jiawei Jiang, Shaoduo Gan, Yue Liu, Fanlin Wang, Gustavo Alonso, Ana Klimovic, Ankit Singla, Wentao Wu, Ce Zhang.
Towards Demystifying Serverless Machine Learning Training. SIGMOD 2021 (to appear).