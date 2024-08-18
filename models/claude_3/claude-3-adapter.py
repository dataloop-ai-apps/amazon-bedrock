import logging
import shutil
import warnings
import torch.backends.cudnn as cudnn
import random
import os
import cv2
import urllib.request
import torch
from glob import glob
import dtlpy as dl
import boto3
import json
from botocore.exceptions import ClientError

logger = logging.getLogger('ModelAdapter')


@dl.Package.decorators.module(description='Model Adapter for Yolovx object detection',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class ModelAdapter(dl.BaseModelAdapter):

    def __init__(self, model_entity: dl.Model):
        # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.client = boto3.client(service_name='bedrock-runtime',
                                   aws_access_key_id="",
                                   aws_secret_access_key="")

        super().__init__(model_entity)

    def load(self, local_path, **kwargs):
        model_filename = os.path.join(local_path, self.configuration.get('weights_filename', 'weights/best_ckpt.pth'))
        checkpoint_url = self.configuration.get('checkpoint_url', None)

        self.model = self.exp.get_model()
        self.model.eval()

        # Load weights from url
        if not os.path.isfile(model_filename):
            if checkpoint_url is not None:
                logger.info("Loading weights from url")
                os.makedirs(local_path, exist_ok=True)
                logger.info("created local_path dir")
                urllib.request.urlretrieve(checkpoint_url,
                                           os.path.join(local_path, self.configuration.get('weights_filename')))
            else:
                raise Exception("checkpoints weights were not loaded! URL not found")

        if os.path.exists(model_filename):
            logger.info("Loading saved weights")
            weights = torch.load(model_filename, map_location=self.device)
            self.model.load_state_dict(weights["model"])
        else:
            raise dl.exceptions.NotFound(f'Model path ({model_filename}) not found! model weights is required')

    @staticmethod
    def move_annotation_files(data_path):
        json_files = glob(os.path.join(data_path, 'json', '**/*.json'))
        json_files += glob(os.path.join(data_path, 'json', '*.json'))

        if os.path.sep == '\\':
            sub_path = '\\'.join(json_files[0].split('json\\')[-1].split('\\')[:-1])
        else:
            sub_path = '/'.join(json_files[0].split('json/')[-1].split('/')[:-1])

        item_files = glob(os.path.join(data_path, 'items', sub_path, '*'))

        for src, dst in zip([json_files, item_files], ['json', 'items']):
            for src_file in src:
                if not os.path.exists(os.path.join(data_path, dst, os.path.basename(src_file))):
                    shutil.move(src_file, os.path.join(data_path, dst, os.path.basename(src_file)))
        for root, dirs, files in os.walk(data_path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)

    def convert_from_dtlpy(self, data_path, **kwargs):
        # Subsets validation
        subsets = self.model_entity.metadata.get("system", dict()).get("subsets", None)
        if 'train' not in subsets:
            raise ValueError('Could not find train set. Yolo-x requires train and validation set for training. '
                             'Add a train set DQL filter in the dl.Model metadata')
        if 'validation' not in subsets:
            raise ValueError('Could not find validation set. Yolo-x requires train and validation set for training. '
                             'Add a validation set DQL filter in the dl.Model metadata')

        for subset, filters_dict in subsets.items():
            filters = dl.Filters(custom_filter=filters_dict)
            filters.add_join(field='type', values='box')
            pages = self.model_entity.dataset.items.list(filters=filters)
            if pages.items_count == 0:
                raise ValueError(f'Could not find box annotations in subset {subset}. '
                                 f'Cannot train without annotation in the data subsets')

        self.move_annotation_files(os.path.join(data_path, 'train'))
        self.move_annotation_files(os.path.join(data_path, 'validation'))

    def predict(self, batch, **kwargs):
        print('predicting batch of size: {}'.format(len(batch)))
        batch_annotations = list()
        for img in batch:
            detections = self.inference(img)
            collection = dl.AnnotationCollection()
            if detections is not None:
                for detection in detections:
                    x0, y0, x1, y1, label, confidence = (detection["x0"], detection["y0"], detection["x1"],
                                                         detection["y1"], detection["label"],
                                                         detection["conf"])

                    collection.add(annotation_definition=dl.Box(left=max(x0, 0),
                                                                top=max(y0, 0),
                                                                right=min(x1, img.shape[1]),
                                                                bottom=min(y1, img.shape[0]),
                                                                label=label
                                                                ),

                                   model_info={'name': self.model_entity.name,
                                               'model_id': self.model_entity.id,
                                               'confidence': float(confidence)})

                batch_annotations.append(collection)

        return batch_annotations

    def inference(self):
        prompt_config = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

        body = json.dumps(prompt_config)

        response_body = None
        results = None

        try:
            response = bedrock.invoke_model(body=body,
                                            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                                            accept="application/json",
                                            contentType="application/json"
                                            )
            response_body = json.loads(response.get("body").read())

        except ClientError as e:
            print("Error: ", e)

        if response_body:
            results = response_body.get("content")[0].get("text")

        return results

    def prepare_item_func(self, item: dl.Item):
        path = item.download(save_locally=True)
        if os.path.exists(path):
            img = cv2.imread(path)
        else:
            img = None
        os.remove(path)
        return img
