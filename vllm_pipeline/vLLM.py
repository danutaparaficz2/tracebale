import base64
import io
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from openai import OpenAI
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
import requests

register_heif_opener()

# ---------------------------------------------------------
# Environment & Path Configuration
# ---------------------------------------------------------

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "token-not-needed")

IMAGE_DIR = "/Users/danuta.paraficz/PyProjects/Tracable/tracebale/Images/"
OUTPUT_DIR = Path("/Users/danuta.paraficz/PyProjects/Tracable/tracebale/vllm_output")
MODEL_NAME = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-VL-32B-Instruct")
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1568"))
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "300"))
SKIP_EXISTING_OUTPUTS = os.getenv("OPENAI_SKIP_EXISTING", "1").strip().lower() in {"1", "true", "yes"}
START_INDEX = max(1, int(os.getenv("OPENAI_START_INDEX", "1")))

HEALTHCHECK_TIMEOUT_SECONDS = float(os.getenv("OPENAI_HEALTHCHECK_TIMEOUT", "15.0"))
HEALTHCHECK_RETRIES = max(1, int(os.getenv("OPENAI_HEALTHCHECK_RETRIES", "3")))
SKIP_ENDPOINT_CHECK = os.getenv("OPENAI_SKIP_ENDPOINT_CHECK", "0").strip().lower() in {"1", "true", "yes"}


# ---------------------------------------------------------
# Utilities & Healthchecks
# ---------------------------------------------------------
def normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
            break
    return normalized


def check_api_endpoint(base_url: str, timeout_seconds: float, retries: int) -> tuple[bool, str]:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return False, f"Invalid OPENAI_BASE_URL: {base_url}"

    last_error = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(f"{base_url}/models", timeout=timeout_seconds)
            response.raise_for_status()
            return True, ""
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < retries:
                print(f"Endpoint check attempt {attempt}/{retries} failed: {exc}. Retrying...")

    return False, f"Cannot reach API endpoint at {base_url} after {retries} attempt(s): {last_error}"


def encode_file_to_data_uri(path: str) -> str:
    """Loads an image, downsizes it (long side <= MAX_IMAGE_SIDE) and encodes it as a JPEG data URI."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert('RGB')
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=90)

    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------
# Inspection Protocol & Prompt (Aligned with SUPSI eBKP-H Schema)
# ---------------------------------------------------------
SYSTEM_PROMPT = """You are an expert structural engineer, circular construction surveyor, and building deconstruction auditor working under Swiss building standards (SIA 430, SNBS, and eBKP-H).
Your objective is to inspect interior and exterior architectural images to identify reusable components, structural/non-structural elements, their condition, disassembly method, and visible degradation anomalies.

Follow these strict auditing criteria:
1. Component Classification (Swiss eBKP-H standard):
   - First level: e.g., C_Costruzione_grezza_edificio, D_Impianti_tecnici_edificio, E_Facciata_edificio, G_Finitura_interna_edificio, I_Esterno_edificio.
   - Second level: e.g., G01_Parete_divisoria_porta_interna_portone_interno, E03_Infisso_in_facciata, D05_Impianto_di_riscaldamento, G04_Rivestimento_di_soffitti.
   - Third level (eBKP-H code): e.g., G0105_Porta_interna, E0301_Finestra, D0505_Diffusione_di_calore, C0302_Pilastro_interno, G0402_Rivestimento_di_soffitti.
2. Condition Assessment:
   - Aesthetics: Strictly choose between: "As new", "Minor signs of deterioration/wear", or "Compromised".
   - Functionality: Operating condition of hardware and assembly (e.g., "Functional", "Functionality does not appear to be compromised", "Damaged / non-functional").
   - Presence of pollutants: Strictly choose between: "Unlikely", "Likely", or "Certain" (e.g., lead-based paint, asbestos core, PCB mastic, sealant residue).
3. Disassembly & Fastening:
   - Connection type: Strictly categorize as: "Connection with added elements (mechanical fasteners)", "Direct integral connection", "Soft chemical connection", or "Hard chemical connection".
   - Difficulty: "Very simple", "Simple", or "Complex".
4. Observed Anomalies:
   - Explicitly list specific visible defects (e.g., "graffiti", "surface scratch / abrasion", "corrosion / rust spot", "paint peeling", "frame deformation").
5. Output format: Pure valid JSON conforming to the requested schema without any markdown fences outside the JSON object."""

USER_PROMPT = """Analyze this image from a building deconstruction audit. Detect all salvageable and architectural components, evaluate their degradation and disassembly traits according to the Swiss eBKP-H circular assessment schema.

Return your response strictly in the following JSON format:

{
  "space_type": "<e.g., circulation corridor, office, stairwell, technical room>",
  "overall_space_condition": "<Good / Fair / Dilapidated>",
  "components": [
    {
      "component_id": "comp_01",
      "ebkp_h_classification": {
        "first_level": "<e.g., G_Finitura_interna_edificio | E_Facciata_edificio | D_Impianti_tecnici_edificio>",
        "second_level": "<e.g., G01_Parete_divisoria_porta_interna_portone_interno | E03_Infisso_in_facciata>",
        "third_level_code": "<e.g., G0105_Porta_interna | E0301_Finestra | D0505_Diffusione_di_calore>"
      },
      "component_name": "<specific element name, e.g., Flush steel fire-rated door leaf | Double-glazed window with PVC frame>",
      "geometry_estimate": "<e.g., 0.90 m x 2.00 m | 1.78 m x 1.50 m | approx. height and width>",
      "conditions": {
        "aesthetics_condition": "<As new | Minor signs of deterioration/wear | Compromised>",
        "functionality_condition": "<Functional | Functionality does not appear to be compromised | Damaged / compromised>",
        "pollutants_flag": "<Unlikely | Likely | Certain>",
        "pollutant_details": "<suspected substance, e.g., lead paint, asbestos core insulation, PCB mastic, none>"
      },
      "disassembly": {
        "connection_type": "<Connection with added elements (mechanical fasteners) | Direct integral connection | Soft chemical connection | Hard chemical connection>",
        "difficulty": "<Very simple | Simple | Complex>",
        "notes": "<practical disassembly guidance, e.g., unbolt hinges first, remove sealant without damaging frame>"
      },
      "observed_anomalies": [
        "<e.g., surface abrasion, graffiti in black marker, minor paint peeling, surface oxidation>"
      ],
      "technical_features": {
        "hardware_present": ["<e.g., black lever handle, mortise lock, door closer>"],
        "labels_or_markings": "<e.g., fire rating EI 60 certification plate, manufacturer badge>"
      },
      "bounding_box_2d": [ymin, xmin, ymax, xmax]
    }
  ]
}"""


def strip_code_fence(text: str) -> str:
    """Removes a surrounding ```json fence if the model added one."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def classify_image(client: OpenAI, image_path: str) -> str:
    """Analyses an image with the prompts and returns the model's JSON text answer."""
    try:
        data_uri = encode_file_to_data_uri(image_path)
    except Exception as e:
        return f"ERROR: {e}"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri,
                                "detail": "high"
                            }
                        },
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ],
        )
        return strip_code_fence(completion.choices[0].message.content or "")

    except Exception as e:
        return f"API_ERROR: {e}"


# ---------------------------------------------------------
# Batch Pipeline
# ---------------------------------------------------------
def main():
    base_url = normalize_base_url(BASE_URL)
    if not SKIP_ENDPOINT_CHECK:
        endpoint_ok, endpoint_error = check_api_endpoint(
            base_url,
            timeout_seconds=HEALTHCHECK_TIMEOUT_SECONDS,
            retries=HEALTHCHECK_RETRIES,
        )
        if not endpoint_ok:
            print(f"ERROR: {endpoint_error}")
            return

    image_dir = Path(IMAGE_DIR)
    if not image_dir.is_dir():
        print(f"ERROR: Directory not found: {IMAGE_DIR}")
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic"}
    image_files = sorted(
        [f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions],
        key=lambda x: x.name.lower()
    )

    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        return

    print(f"Found {len(image_files)} images in {IMAGE_DIR}")

    client = OpenAI(
        api_key=API_KEY,
        base_url=base_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(image_files, 1):
        if i < START_INDEX:
            continue
        print(f"[{i}/{len(image_files)}] Analysing {image_path.name}...", flush=True)
        output = OUTPUT_DIR / f"{image_path.stem}.json"
        if SKIP_EXISTING_OUTPUTS and output.exists():
            try:
                json.loads(output.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
            else:
                print(f"[{i}/{len(image_files)}] Skipping existing output {output.name}", flush=True)
                continue

        text = classify_image(client, str(image_path))

        if text.startswith("API_ERROR:"):
            print(text)
            print("Aborting to avoid sending queries to a failing endpoint.")
            return

        print(text, end="\n\n")
        output.write_text(text, encoding="utf-8")

    print(f"Processing complete. Text results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()