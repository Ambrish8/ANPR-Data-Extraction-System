import re
import cv2
import easyocr
import numpy as np
import Levenshtein
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ===========================
# OCR MODEL INITIALIZATION
# ===========================

DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

print("Loading EasyOCR...")
easy_reader = easyocr.Reader(
    ['en'],
    gpu=(DEVICE == "cuda"),
    verbose=False
)

print("Loading TrOCR...")
trocr_model_name = "microsoft/trocr-base-printed"

trocr_processor = TrOCRProcessor.from_pretrained(trocr_model_name)

trocr_model = VisionEncoderDecoderModel.from_pretrained(
    trocr_model_name
).to(DEVICE)

trocr_model.config.max_length = 64
trocr_model.config.num_beams = 4


# ===========================
# TEXT CLEANING
# ===========================

def clean_plate_text(text):
    if text is None:
        return ""

    s = " ".join(str(text).split())
    s = s.upper()
    s = re.sub(r"[^A-Z0-9 \-]", "", s)

    return s


# ===========================
# IMAGE PREPROCESSING
# ===========================

def enhance_contrast_clahe(gray):
    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )
    return clahe.apply(gray)


def unsharp_mask(img, radius=5, amount=1.5):
    blurred = cv2.GaussianBlur(img, (0, 0), radius)

    sharpened = cv2.addWeighted(
        img,
        1 + amount,
        blurred,
        -amount,
        0,
    )

    return sharpened


def denoise(img):
    return cv2.fastNlMeansDenoising(
        img,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def adaptive_preprocess(bgr):

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    if max(h, w) < 300:
        gray = cv2.resize(
            gray,
            (w * 2, h * 2),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = denoise(gray)

    gray = enhance_contrast_clahe(gray)

    gray = unsharp_mask(
        gray,
        radius=1,
        amount=1.2,
    )

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        10,
    )

    return th


# ===========================
# EASY OCR
# ===========================

def run_easyocr_on_image(img):

    try:

        res = easy_reader.readtext(
            img,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )

        if not res:
            return "", 0.0

        texts, confs = zip(
            *[(r[1].upper(), r[2]) for r in res]
        )

        txt = re.sub(
            r"[^A-Z0-9]",
            "",
            "".join(texts),
        )

        return txt, float(np.mean(confs))

    except:
        return "", 0.0


# ===========================
# TrOCR
# ===========================

def run_trocr_on_image(img):

    try:

        if isinstance(img, np.ndarray):
            rgb = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB,
            )
            pil_img = Image.fromarray(rgb)

        else:
            pil_img = img

        pixel_values = trocr_processor(
            images=pil_img,
            return_tensors="pt",
        ).pixel_values.to(DEVICE)

        outputs = trocr_model.generate(
            pixel_values,
            return_dict_in_generate=True,
            output_scores=True,
        )

        pred = trocr_processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
        )[0]

        pred = re.sub(
            r"[^A-Z0-9]",
            "",
            pred.upper(),
        )

        scores = outputs.scores

        if not scores:
            conf = 0.0

        else:

            token_probs = []

            for step_logits in scores:

                logits = step_logits[0].detach().cpu().numpy()

                exp = np.exp(
                    logits - np.max(logits)
                )

                probs = exp / exp.sum()

                token_probs.append(np.max(probs))

            conf = float(
                np.exp(
                    np.mean(
                        np.log(
                            np.clip(
                                token_probs,
                                1e-9,
                                1.0,
                            )
                        )
                    )
                )
            )

        return pred, conf

    except:
        return "", 0.0


# ===========================
# FORMAT CORRECTION
# ===========================

LETTER_TO_DIGIT = {
    "O": "0",
    "I": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
    "T": "7",
    "A": "4",
    "P": "9",
    "Q": "0",
    "D": "0",
    "U": "0",
    "J": "3",
    "E": "3",
    "F": "7",
    "C": "0",
    "K": "4",
    "L": "1",
    "H": "4",
    "M": "0",
    "N": "0",
    "R": "8",
    "V": "0",
    "W": "0",
    "X": "0",
    "Y": "4",
}

DIGIT_TO_LETTER = {
    v: k
    for k, v in LETTER_TO_DIGIT.items()
    if v.isdigit() and k.isalpha()
}

FMT_LONG = "LLDDLLDDDD"
FMT_SHORT = "LLDDLDDDD"


def enforce_plate_format_strict(candidate):

    cand = re.sub(
        r"[^A-Z0-9]",
        "",
        candidate.upper(),
    )

    if not cand:
        return "", False

    def fix(text, pattern):

        if len(text) != len(pattern):
            return text, float("inf")

        out = list(text)

        cost = 0

        for i, (ch, req) in enumerate(zip(text, pattern)):

            if req == "L":

                if not ch.isalpha():

                    if ch in DIGIT_TO_LETTER:
                        out[i] = DIGIT_TO_LETTER[ch]
                        cost += 1

            else:

                if not ch.isdigit():

                    if ch.upper() in LETTER_TO_DIGIT:
                        out[i] = LETTER_TO_DIGIT[ch.upper()]
                        cost += 1

        return "".join(out), cost

    corr10, c10 = fix(cand, FMT_LONG)

    ok10 = (
        c10 != float("inf")
        and re.fullmatch(
            r"[A-Z]{2}\d{2}[A-Z]{2}\d{4}",
            corr10,
        )
    )

    corr9, c9 = fix(cand, FMT_SHORT)

    ok9 = (
        c9 != float("inf")
        and re.fullmatch(
            r"[A-Z]{2}\d{2}[A-Z]\d{4}",
            corr9,
        )
    )

    if ok10 and (c10 <= c9 or not ok9):
        return corr10, True

    if ok9:
        return corr9, True

    return cand, False


# ===========================
# ENSEMBLE
# ===========================

def ensemble_decide(
    easy_raw,
    easy_proc,
    trocr_raw,
    trocr_proc,
):

    entries = [
        (easy_raw[0], easy_raw[1]),
        (easy_proc[0], easy_proc[1]),
        (trocr_raw[0], trocr_raw[1]),
        (trocr_proc[0], trocr_proc[1]),
    ]

    entries = [
        (t, c)
        for t, c in entries
        if t
    ]

    if not entries:
        return "", 0.0

    scores = {}

    for txt, conf in entries:

        txt = re.sub(
            r"[^A-Z0-9]",
            "",
            txt.upper(),
        )

        scores.setdefault(txt, []).append(conf)

    best = max(
        scores,
        key=lambda x: np.mean(scores[x]),
    )

    final_plate, _ = enforce_plate_format_strict(best)

    final_conf = np.mean(scores[best])

    return final_plate, final_conf
