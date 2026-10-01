"""
GPT Image 2.5 Sunburst Edit Node (fal.ai)
Custom node de ComfyUI para el endpoint:
    openai/gpt-image-2.5/sunburst/edit

- Edicion con hasta 16 imagenes de referencia (image_urls)
- Mascara opcional (mask_url)
- API key manual en el propio nodo (o env FAL_KEY / config.ini)
"""

import os
import io
import base64
import tempfile
import configparser

import torch
import numpy as np
import requests
from PIL import Image
from fal_client.client import SyncClient


ENDPOINT = "openai/gpt-image-2.5/sunburst/edit"
MAX_REFERENCE_IMAGES = 16
KEY_PLACEHOLDERS = {"", "<your_fal_api_key_here>", "YOUR_FAL_KEY"}


# ==================== API KEY ====================

def resolve_api_key(node_key=""):
    """
    Prioridad:
      1. Campo 'api_key' del nodo
      2. Variable de entorno FAL_KEY
      3. config.ini (junto a este archivo o en la carpeta padre)
    """
    if node_key and node_key.strip() not in KEY_PLACEHOLDERS:
        return node_key.strip()

    env_key = os.environ.get("FAL_KEY", "").strip()
    if env_key not in KEY_PLACEHOLDERS:
        return env_key

    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here, os.path.dirname(here)):
        path = os.path.join(folder, "config.ini")
        if os.path.isfile(path):
            cfg = configparser.ConfigParser()
            cfg.read(path)
            cfg_key = cfg.get("API", "FAL_KEY", fallback="").strip()
            if cfg_key not in KEY_PLACEHOLDERS:
                return cfg_key

    raise ValueError(
        "No se encontro la FAL API key. Pegala en el campo 'api_key' del nodo, "
        "defínela como variable de entorno FAL_KEY o ponla en config.ini. "
        "Consigue una en https://fal.ai/dashboard/keys"
    )


# ==================== IMAGE UTILS ====================

def tensor_to_pil_list(image):
    """Convierte un tensor IMAGE de ComfyUI [B,H,W,C] en una lista de PIL Images."""
    arr = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.array(image)
    if arr.ndim == 3:
        arr = arr[None]
    frames = []
    for frame in arr:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        frames.append(Image.fromarray(frame))
    return frames


def mask_to_pil(mask, size, invert=False):
    """Convierte un MASK de ComfyUI en una imagen PIL en escala de grises."""
    arr = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
    while arr.ndim > 2:
        arr = arr[0]
    if invert:
        arr = 1.0 - arr
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
    if img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return img


def upload_pil(client, pil_image):
    """Sube una imagen PIL a fal.ai y devuelve la URL."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_image.save(tmp, format="PNG")
            tmp_path = tmp.name
        return client.upload_file(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def download_image(url):
    """Descarga una imagen (URL normal o data URI) y devuelve un PIL Image."""
    if url.startswith("data:"):
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))


def pil_to_comfy(pil_img):
    """Devuelve (rgb_array, mask_array) en float32 0-1. mask = 1 - alpha (convencion LoadImage)."""
    has_alpha = pil_img.mode in ("RGBA", "LA") or (
        pil_img.mode == "P" and "transparency" in pil_img.info
    )
    if has_alpha:
        rgba = np.array(pil_img.convert("RGBA")).astype(np.float32) / 255.0
        return rgba[..., :3], 1.0 - rgba[..., 3]
    rgb = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    return rgb, np.zeros(rgb.shape[:2], dtype=np.float32)


# ==================== NODO ====================

class GPTImage25SunburstEdit:
    """
    GPT Image 2.5 Sunburst Edit
    Edita imagenes con OpenAI GPT Image 2.5 (variante Sunburst) via fal.ai.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Imagen principal a editar (si es un batch se envian todos los frames)"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Instruccion de edicion"}),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Tu FAL API key. Si lo dejas vacio se usa FAL_KEY del entorno o config.ini. "
                               "OJO: lo que escribas aqui se guarda en el workflow (.json).",
                }),
            },
            "optional": {
                "image_2": ("IMAGE", {"tooltip": "Imagen de referencia 2"}),
                "image_3": ("IMAGE", {"tooltip": "Imagen de referencia 3"}),
                "image_4": ("IMAGE", {"tooltip": "Imagen de referencia 4"}),
                "mask": ("MASK", {"tooltip": "Mascara opcional: indica que zona de la imagen editar"}),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "Invierte la mascara antes de enviarla"}),
                "image_size_mode": (["preset", "custom"], {"default": "preset"}),
                "image_size_preset": (
                    ["auto", "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"],
                    {"default": "auto", "tooltip": "'auto' infiere el tamano desde las imagenes de entrada"},
                ),
                "custom_width": ("INT", {"default": 1280, "min": 256, "max": 3840, "step": 16, "tooltip": "Solo en modo custom (multiplo de 16)"}),
                "custom_height": ("INT", {"default": 720, "min": 256, "max": 3840, "step": 16, "tooltip": "Solo en modo custom (multiplo de 16)"}),
                "quality": (["auto", "low", "medium", "high", "xhigh", "max"], {"default": "high", "tooltip": "Mas calidad = mas detalle, latencia y coste"}),
                "background": (["auto", "transparent", "opaque"], {"default": "auto"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "output_compression": ("INT", {"default": -1, "min": -1, "max": 100, "tooltip": "0-100, solo jpeg/webp. -1 = no enviar"}),
                "sync_mode": ("BOOLEAN", {"default": False, "tooltip": "Devuelve data URI; no queda en el historial de fal"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha_mask")
    FUNCTION = "edit"
    CATEGORY = "plexard_FAL"
    DESCRIPTION = "GPT Image 2.5 Sunburst Edit (fal.ai) - edicion con multiples referencias y mascara opcional"

    def edit(
        self,
        image,
        prompt,
        api_key="",
        image_2=None,
        image_3=None,
        image_4=None,
        mask=None,
        invert_mask=False,
        image_size_mode="preset",
        image_size_preset="auto",
        custom_width=1280,
        custom_height=720,
        quality="high",
        background="auto",
        num_images=1,
        output_format="png",
        output_compression=-1,
        sync_mode=False,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("El prompt no puede estar vacio.")

        key = resolve_api_key(api_key)
        client = SyncClient(key=key)

        # ---- Recopilar y subir imagenes ----
        pil_images = []
        for tensor in (image, image_2, image_3, image_4):
            if tensor is not None:
                pil_images.extend(tensor_to_pil_list(tensor))

        if len(pil_images) > MAX_REFERENCE_IMAGES:
            print(f"[GPTImage25] Aviso: maximo {MAX_REFERENCE_IMAGES} imagenes, se ignoran {len(pil_images) - MAX_REFERENCE_IMAGES}.")
            pil_images = pil_images[:MAX_REFERENCE_IMAGES]

        image_urls = [upload_pil(client, p) for p in pil_images]
        if not image_urls or not all(image_urls):
            raise RuntimeError("No se pudieron subir las imagenes a fal.ai.")

        # ---- Argumentos ----
        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "quality": quality,
            "background": background,
            "num_images": int(num_images),
            "output_format": output_format,
        }

        if image_size_mode == "custom":
            w = max(256, min(3840, (int(custom_width) // 16) * 16))
            h = max(256, min(3840, (int(custom_height) // 16) * 16))
            total = w * h
            if not (655_360 <= total <= 8_294_400) or max(w / h, h / w) > 3:
                print("[GPTImage25] Aviso: tamano fuera de los limites documentados "
                      "(655.360-8.294.400 px, ratio <= 3:1). La API podria rechazarlo.")
            arguments["image_size"] = {"width": w, "height": h}
        else:
            arguments["image_size"] = image_size_preset

        if output_format in ("jpeg", "webp") and output_compression >= 0:
            arguments["output_compression"] = int(output_compression)

        if sync_mode:
            arguments["sync_mode"] = True

        if mask is not None:
            mask_pil = mask_to_pil(mask, pil_images[0].size, invert=invert_mask)
            arguments["mask_url"] = upload_pil(client, mask_pil)

        # ---- Llamada a la API ----
        def on_update(update):
            logs = getattr(update, "logs", None)
            if logs:
                for log in logs:
                    msg = log.get("message") if isinstance(log, dict) else str(log)
                    if msg:
                        print(f"[GPTImage25] {msg}")

        try:
            result = client.subscribe(
                ENDPOINT,
                arguments=arguments,
                with_logs=True,
                on_queue_update=on_update,
            )
        except Exception as e:
            raise RuntimeError(f"Error en fal.ai ({ENDPOINT}): {e}") from e

        # ---- Procesar resultado ----
        images_info = result.get("images") or []
        if not images_info:
            raise RuntimeError(f"La API no devolvio imagenes. Respuesta: {result}")

        rgb_list, mask_list = [], []
        for info in images_info:
            rgb, alpha_mask = pil_to_comfy(download_image(info["url"]))
            rgb_list.append(rgb)
            mask_list.append(alpha_mask)

        # Por si algun resultado viniera con distinto tamano
        h0, w0 = rgb_list[0].shape[:2]
        for i, (rgb, am) in enumerate(zip(rgb_list, mask_list)):
            if rgb.shape[:2] != (h0, w0):
                rgb_list[i] = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((w0, h0), Image.LANCZOS)).astype(np.float32) / 255.0
                mask_list[i] = np.array(Image.fromarray((am * 255).astype(np.uint8)).resize((w0, h0), Image.BILINEAR)).astype(np.float32) / 255.0

        image_tensor = torch.from_numpy(np.stack(rgb_list, axis=0))
        mask_tensor = torch.from_numpy(np.stack(mask_list, axis=0))
        return (image_tensor, mask_tensor)


# Mapeo para ComfyUI
NODE_CLASS_MAPPINGS = {
    "GPTImage25SunburstEdit": GPTImage25SunburstEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImage25SunburstEdit": "GPT Image 2.5 Sunburst Edit",
}
