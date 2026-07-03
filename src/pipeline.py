import os
import re
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path

from model_inference import yolo
from text_processing import (
    clean_plate_text,
    adaptive_preprocess,
    run_easyocr_on_image,
    run_trocr_on_image,
    ensemble_decide,
)

# ==========================
# PATHS (modify as required)
# ==========================

XML_DIR = "data/xml"
TEST_IMG_DIR = "data/images"
OUTPUT_DIR = "output"

GROUND_TRUTH_CSV = os.path.join(
    OUTPUT_DIR,
    "ground_truth.csv"
)

N_IMAGES = 100
N_VIS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# XML READING
# ==========================

def extract_from_xml(xml_path):

    try:

        tree = ET.parse(xml_path)

        root = tree.getroot()

        name_elem = root.find(".//name")

        if name_elem is not None and name_elem.text:

            return clean_plate_text(name_elem.text)

        return ""

    except Exception:

        return ""

# ==========================
# GROUND TRUTH CSV
# ==========================

xml_files = sorted(Path(XML_DIR).glob("*.xml"))

gt_results = []

for xml_path in xml_files:

    gt = extract_from_xml(xml_path)

    gt_results.append(
        {
            "image_name": xml_path.stem,
            "xml_name": xml_path.name,
            "ground_truth": gt,
        }
    )

gt_df = pd.DataFrame(gt_results)

gt_df.to_csv(
    GROUND_TRUTH_CSV,
    index=False
)

print("Ground Truth CSV Saved")

# ==========================
# PROCESS IMAGES
# ==========================

files = [
    f
    for f in os.listdir(TEST_IMG_DIR)
    if f.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png",
        )
    )
]

sample_files = random.sample(
    files,
    min(
        N_IMAGES,
        len(files),
    ),
)

results = []

for idx, fname in enumerate(sample_files, 1):

    print(f"[{idx}/{len(sample_files)}] {fname}")

    img_path = os.path.join(
        TEST_IMG_DIR,
        fname,
    )

    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        continue

    yres = yolo.predict(
        source=img_path,
        conf=0.5,
        verbose=False,
    )

    if (
        len(yres) == 0
        or not hasattr(yres[0], "boxes")
    ):

        results.append(
            {
                "Image": fname,
                "Detected": False,
            }
        )

        continue

    boxes = yres[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:

        results.append(
            {
                "Image": fname,
                "Detected": False,
            }
        )

        continue

    x1, y1, x2, y2 = boxes[0].astype(int)

    crop = img_bgr[
        y1:y2,
        x1:x2,
    ]

    processed = adaptive_preprocess(crop)

    easy_raw = run_easyocr_on_image(crop)

    easy_proc = run_easyocr_on_image(processed)

    trocr_raw = run_trocr_on_image(crop)

    processed_bgr = cv2.cvtColor(
        processed,
        cv2.COLOR_GRAY2BGR,
    )

    trocr_proc = run_trocr_on_image(
        processed_bgr
    )

    final_text, final_conf = ensemble_decide(
        easy_raw,
        easy_proc,
        trocr_raw,
        trocr_proc,
    )

    results.append(
        {
            "Image": fname,

            "Easy_Raw": easy_raw[0],
            "Easy_Raw_Conf": easy_raw[1],

            "Easy_Proc": easy_proc[0],
            "Easy_Proc_Conf": easy_proc[1],

            "TrOCR_Raw": trocr_raw[0],
            "TrOCR_Raw_Conf": trocr_raw[1],

            "TrOCR_Proc": trocr_proc[0],
            "TrOCR_Proc_Conf": trocr_proc[1],

            "Final": final_text,
            "Final_Conf": final_conf,

            "Detected": True,
        }
    )

df = pd.DataFrame(results)

csv_path = os.path.join(
    OUTPUT_DIR,
    "ensemble_easy_trocr.csv",
)

df.to_csv(
    csv_path,
    index=False,
)

print("Prediction CSV Saved")
import Levenshtein

# ==========================
# METRICS
# ==========================

merged = pd.merge(
    df,
    gt_df,
    left_on=df.Image.str.replace(r"\.[^.]+$", "", regex=True),
    right_on="image_name",
    how="left",
)


def normalized_similarity(a, b):

    if pd.isna(a) or pd.isna(b):
        return 0.0

    if len(a) == 0 and len(b) == 0:
        return 1.0

    return 1 - (
        Levenshtein.distance(a, b)
        / max(len(a), len(b))
    )


merged["Similarity"] = merged.apply(
    lambda row: normalized_similarity(
        str(row["Final"]),
        str(row["ground_truth"]),
    ),
    axis=1,
)

merged["Exact_Match"] = (
    merged["Final"] == merged["ground_truth"]
)

metrics = {

    "Total Images":
        len(merged),

    "Detected":
        merged["Detected"].sum(),

    "Detection Rate (%)":
        merged["Detected"].mean() * 100,

    "Exact Match (%)":
        merged["Exact_Match"].mean() * 100,

    "Average Similarity":
        merged["Similarity"].mean(),

    "Average Confidence":
        merged["Final_Conf"].mean(),
}

print("\n========== METRICS ==========\n")

for k, v in metrics.items():

    print(f"{k:25s}: {v}")

metrics_path = os.path.join(
    OUTPUT_DIR,
    "metrics.csv",
)

merged.to_csv(
    metrics_path,
    index=False,
)

print("\nMetrics CSV Saved")


# ==========================
# VISUALIZATION
# ==========================

vis = merged.sample(
    min(N_VIS, len(merged))
).reset_index(drop=True)

plt.figure(
    figsize=(18, 4 * len(vis))
)

for i, row in vis.iterrows():

    img_path = os.path.join(
        TEST_IMG_DIR,
        row.Image,
    )

    img_bgr = cv2.imread(img_path)

    yres = yolo.predict(
        source=img_path,
        conf=0.5,
        verbose=False,
    )

    if (
        len(yres) == 0
        or not hasattr(yres[0], "boxes")
    ):
        continue

    boxes = yres[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        continue

    x1, y1, x2, y2 = boxes[0].astype(int)

    crop = img_bgr[
        y1:y2,
        x1:x2,
    ]

    processed = adaptive_preprocess(crop)

    # --------------------------

    ax0 = plt.subplot(
        len(vis),
        4,
        i * 4 + 1,
    )

    display = img_bgr.copy()

    cv2.rectangle(
        display,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        display,
        row.Final,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    ax0.imshow(
        cv2.cvtColor(
            display,
            cv2.COLOR_BGR2RGB,
        )
    )

    ax0.set_title(
        f"{row.Image}\n"
        f"GT : {row.ground_truth}\n"
        f"Pred : {row.Final}\n"
        f"Conf : {row.Final_Conf:.2f}"
    )

    ax0.axis("off")

    # --------------------------

    ax1 = plt.subplot(
        len(vis),
        4,
        i * 4 + 2,
    )

    ax1.imshow(
        cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB,
        )
    )

    ax1.set_title("Plate Crop")

    ax1.axis("off")

    # --------------------------

    ax2 = plt.subplot(
        len(vis),
        4,
        i * 4 + 3,
    )

    ax2.imshow(
        processed,
        cmap="gray",
    )

    ax2.set_title("Processed")

    ax2.axis("off")

    # --------------------------

    ax3 = plt.subplot(
        len(vis),
        4,
        i * 4 + 4,
    )

    txt = (
        f"Easy Raw   : {row.Easy_Raw}\n"
        f"Easy Proc  : {row.Easy_Proc}\n"
        f"TrOCR Raw  : {row.TrOCR_Raw}\n"
        f"TrOCR Proc : {row.TrOCR_Proc}\n\n"
        f"Ground Truth : {row.ground_truth}\n"
        f"Prediction   : {row.Final}\n"
        f"Similarity   : {row.Similarity:.3f}"
    )

    ax3.text(
        0,
        0.5,
        txt,
        fontsize=10,
        fontfamily="monospace",
    )

    ax3.axis("off")

plt.tight_layout()

plt.show()

print("\nPipeline Finished Successfully.")
