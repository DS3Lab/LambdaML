
# $\lambda$-ML

$\lambda$-ML is a machine learning system built in serverless infrastructure (Amazon AWS Lambda).
Serverless compute service lets you run code without provisioning or managing servers, creating workload-aware cluster scaling logic, maintaining event integrations, or managing runtimes. 
Different from VM-based cloud compute, compute instances in serverless infrastructure cannot communicate with each other.
To solve this problem, $\lambda$-ML implements various communication patterns using external storage.


## Video Tutorials

We provide several video tutorials on YouTube.

- [Introduction to $\lambda$-ML](https://youtu.be/yUzzdp4IY7k)
- [Programming Interface](https://youtu.be/YU0974fViSU)
- [Deploying $\lambda$-ML with S3](https://youtu.be/E_kzXTm32EM)
- [Deploying $\lambda$-ML with ElastiCache](https://youtu.be/58PMo2N8rxA)
- [Deploying $\lambda$-ML with DynamoDB](https://youtu.be/mWa3NpCcEDU)
- [Deploying $\lambda$-ML with Hybrid Parameter Server](https://youtu.be/gjmEV0RCaak)

### Development notes
- Two AWS Lambda functions are needed - Trigger and Execution function. Trigger function is used to invoke the workers, which then execute the code of the Execution Function. The Trigger function requires the name of the Execution function in order to invoke it.
- The only functionality that Trigger Function needs to be able to perform is to invoke AWS Lambda Functions. If it is not in the VPC, the execution role needs to have permissions to invoke Lambdas. If it is in the VPC, it also needs a VPC Endpoint Interface to access AWS Lambda.
- Regardless of the choice for storage (S3, Elasticache or DynamoDB), Execution function needs to be able to access S3 buckets. If it is not in the VPC, the execution role needs to have permissions to access S3. If it is in the VPC, it also needs a VPC Endpoint Gateway to access S3.
- Since Elasticache is designed to be used internally inside a VPC, if Memcached or Redis are your choice for storage, you need to have your Execution function inside a VPC.
- Due to the size requirement for the code inside a Lambda Function (50MB), you can't zip the whole LambdaML and upload it to the function. Instead, only zip the packages that you need for your project.
- In the case of Memcached, you need to change the maximum file size in the parameter group. Consider watching the ElastiCache video for more details.


## Dependencies
- awscli (version 1)
- botocore
- boto3
- numpy
- torch=1.0.1
- thrift
- redis
- grpcio

## Environment setup

- Create a Lambda layer with PyTorch 1.0.1.
- Compress the whole project and upload to Lambda.
- Create a VPC and a security group in AWS.

## Programming Interface

$\lambda$-ML leverages external storage services, e.g., S3, Elasticache, and DynamoDB, to implement communication between serverless compute instances.
We provide both storage interfaces and communication primitives.

### Storage

The storage layer offers basic operations to manipulate external storage.

- S3 ([storage/s3/s3_type.py](storage/s3/s3_type.py)). storage operations: list/save/load/delete/clear/...
- Elasticache ([storage/memcached/memcached_type.py](storage/memcached/memcached_type.py)). storage operations: list/save/load/delete/clear/...
- DynamoDB ([storage/dynamo/dynamo_type.py](storage/dynamo/dynamo_type.py)). storage operations: list/save/load/delete/clear/...

### Communication primitive

The communication layer provides popular communication primitives.

- S3 communicator ([communicator/s3_comm.py](communicator/s3_comm.py)). primitives: async/reduce/reduce_scatter.
- Elasticache communicator ([communicator/memcached_comm.py](communicator/memcached_comm.py)). primitives: async/reduce/reduce_scatter.
- DynamoDB communicator ([communicator/dynamo_comm.py](communicator/dynamo_comm.py)). primitives: async/reduce/reduce_scatter.

### Hybrid framework.

In addition to storage services, $\lambda$-ML also implements a hybrid architecture ---
one VM acts as a parameter server and serverless instances communicate with the VM.
- Launch parameter server. see [thrift_ps/start_service.py](thrift_ps/start_service.py)
- Communication interfaces: ping/register/pull/push/delete.

## Usage

The general usage of $\lambda$-ML:
1. Partition the dataset and upload to S3.
2. Create a trigger Lambda function and an execution Lambda function.
3. Set configurations (e.g., dataset location) and hyperparameters (e.g., learning rate).
4. Set VPC and security group.
5. Execute the trigger function.
6. See the logs in CloudWatch.

See [examples](reproducibility.md) for more details.


## Contact

If you have any question or suggestion, feel free to contact jiawei.jiang@inf.ethz.ch and ce.zhang@inf.ethz.ch.


## Reference
Jiawei Jiang, Shaoduo Gan, Yue Liu, Fanlin Wang, Gustavo Alonso, Ana Klimovic, Ankit Singla, Wentao Wu, Ce Zhang.
[Towards Demystifying Serverless Machine Learning Training](https://arxiv.org/abs/2105.07806). *SIGMOD* 2021.
