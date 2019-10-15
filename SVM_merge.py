{\rtf1\ansi\ansicpg1252\cocoartf1671\cocoasubrtf500
{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset134 PingFangSC-Regular;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import json\
import boto3\
from numpy import vstack, mean\
from torch import load, save\
from urllib.parse import unquote_plus\
\
class s3_handler():\
    def __init__(self):\
        self.client = boto3.client('s3')\
        \
    def get_data(self, object_key, bucket_name):\
        client = self.client\
        download_path = '/tmp/\{\}'.format(object_key)\
        client.download_file(bucket_name, object_key, download_path)\
        self.bucket = bucket_name\
        return(download_path)\
        \
    def send_data(self, file_name, upload_path):\
        # Put the object\
        client = self.client\
        client.upload_file(upload_path,self.bucket, file_name)\
        return True\
\
model_file = "target_model.pt"\
\
def handler(event, context):\
    #respond to the invokation\
    record = event['Records'][0]\
    bucket = record['s3']['bucket']['name']\
    object_key = unquote_plus(record['s3']['object']['key'])\
    \
    #convergence parameter\
    eps = 1e-1\
    \
    #s3.operator\
    s3 = s3_handler()\
    \
    #get batcht info\
    ref_batch = load(s3.get_data(object_key,bucket))\
    batch_model = ref_batch['model']\
    batch_cost = ref_batch['cost']\
    \
    #get overall info\
    ref_model = load(s3.get_data(model_file,bucket))\
    index = ref_model['index']\
    previous_model = ref_model['model']\
    previous_cost = ref_model['cost']\
    if index != 6: \
        index = index+1\
        target_model = \{'index':index,'model':vstack((previous_model,batch_model)),'cost':vstack((previous_cost,batch_cost))\}\
        print(index)\
    else:\
        index = 100 # just to get it work for one round\
        target_model = \{'index':index,'model':mean(previous_model[1:6],axis=0),'cost':mean(previous_cost[1:6],axis=0)\}\
        print("hahah")\
        #deciding wether convergent or not\
        try:\
            cost = previous_cost[0]\
        except:\
            cost = previous_cost\
        """\
        if abs(cost-target_model['cost']) <= eps:\
            index = 0\
        """\
    \
    file_name = "target_model.pt"\
    upload_path = '/tmp/\{\}'.format(file_name)\
    save(target_model,upload_path)\
    s3.send_data(file_name,upload_path)\
    \
    if index == 6 :\
        index = 0\
        #
\f1 \'b5\'f7\'d3\'c3
\f0 svm_trigger
\f1 \'ca\'b9\'b5\'c3
\f0 invoke batch analysis\
        lambda_client = boto3.client('lambda')\
        status = lambda_client.invoke(FunctionName='SVM_trigger',InvocationType='RequestResponse')\
    \
    return \{\
        'statusCode': 200,\
        'body': json.dumps('emergence')\
    \}\
}