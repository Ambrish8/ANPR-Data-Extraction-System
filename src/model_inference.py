import os, random, re, math, csv, xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import torch
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import Levenshtein

YOLO_WEIGHTS = "/content/best.pt"
TEST_IMG_DIR = "/content/drive/MyDrive/datas2/images"
XML_DIR = "/content/drive/MyDrive/datas2/images"
OUTPUT_DIR = "/content/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_IMAGES = 300
N_VIS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GROUND_TRUTH_CSV = os.path.join(OUTPUT_DIR, "ground_truth.csv")

print("Loading YOLO...")
yolo = YOLO(YOLO_WEIGHTS)
