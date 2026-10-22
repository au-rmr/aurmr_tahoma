import io
import os
import re
import json
import torch
import html
import base64
import itertools
import h5py

import numpy as np
import supervision as sv
import matplotlib.pyplot as plt

from IPython import display
from IPython.core.display import HTML
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoProcessor,
    get_scheduler
)
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Generator
from peft import LoraConfig, get_peft_model
from PIL import Image
from roboflow import Roboflow
from datetime import datetime


from aurmr_setup.utils.workspace_utils import get_active_workspace_path


task_prompt3 = '<REGION_PROPOSAL>'

workspace_path = get_active_workspace_path()
#CHECKPOINT = "microsoft/Florence-2-base-ft"
CHECKPOINT = f'{workspace_path}/src/VLM_Florence//model_checkpoints/trained/Region_Proposal/epoch_1_2024-07-18-19-03-11'

REVISION = 'refs/pr/6'


class FlorenceModel:

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None

    def build(self, device=None):

        if not device:
            #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device = "cpu"
        self.device = device
        model_params = {"trust_remote_code": True, "revision": REVISION}
        self.model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, **model_params).to(device)
        self.processor = AutoProcessor.from_pretrained(CHECKPOINT, **model_params)


    @torch.no_grad()
    def florence2(self, task_prompt, images):
        image = images[0]


        image_size = (image.shape[1], image.shape[0])
        print('image_size: ',image_size)


        image = torch.from_numpy(image)
        image = image.to(self.device)


        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids,skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=image_size)
        # print('parsed_answer: ',parsed_answer)

        return parsed_answer


    def run(self, images):

        task_prompt = "<OD>"
        task_prompt = "<REGION_PROPOSAL>"



        images[0] = images[0][..., ::-1].copy()

        response = self.florence2(task_prompt, images)

        data = response[task_prompt]

        if True:

            image = images[0]

            detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)

            bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
            label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

            image = bounding_box_annotator.annotate(image, detections)
            image = label_annotator.annotate(image, detections)
            #image.thumbnail((600, 600))

            Image.fromarray(image).save("/tmp/debug.png")



        return data