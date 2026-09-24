"""
Per-image descriptive captioning using BLIP (Salesforce
blip-image-captioning-large, ~1.9GB, auto-downloaded via
transformers/huggingface_hub on first use, cached locally).

This exists to fix a real flaw found in this project's earlier Flux.2
pipeline and initially present in this Krea-2 pipeline too: captions
generated purely by templating a detected face-angle bucket
("frontal", "three_quarter_left", ...) collapse a whole dataset down
to as few as 5 distinct caption strings, repeated verbatim across
every image that happens to share a pose bucket. That is not a
description of the image -- it's a bucket label, and it teaches the
model to associate the trigger token with "images labeled X" rather
than with an actual rich correlation between pixels and words.

This module looks at each image's actual pixels and generates a real,
per-image natural-language description, which is then combined with
the trigger token and framing metadata (see build_full_caption) --
NOT used to replace/discard that metadata, but to add real per-image
variation that the framing-bucket approach structurally cannot provide.
"""
import threading

_MODEL_LOCK = threading.Lock()
_processor = None
_model = None
_device = None


def _lazy_load():
    global _processor, _model, _device
    if _model is not None:
        return
    with _MODEL_LOCK:
        if _model is not None:
            return
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "Salesforce/blip-image-captioning-large"
        print(f"Loading captioning model ({model_id}) on {_device}... (first run downloads ~1.9GB, cached after)")
        _processor = BlipProcessor.from_pretrained(model_id)
        _model = BlipForConditionalGeneration.from_pretrained(model_id).to(_device)
        _model.eval()


def describe_image(pil_image):
    """
    Returns a short natural-language description of what BLIP actually
    sees in this specific image (e.g. "a woman in a blue jacket standing
    on a city street holding a coffee cup"). This is real per-image
    content, not a templated pose/framing bucket label.
    """
    _lazy_load()
    import torch

    inputs = _processor(images=pil_image, return_tensors="pt").to(_device)
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=40, num_beams=3)
    caption = _processor.decode(out[0], skip_special_tokens=True).strip()
    return caption


def build_full_caption(trigger, class_noun, pose, shot_type, pil_image, style_hint=""):
    """
    Combines:
      - the trigger token (identity anchor -- must appear in every caption
        so the model learns to bind it to the subject)
      - a real, per-image BLIP description of the actual photo content
        (clothing, setting, action -- varies per image, unlike a pose bucket)

    Deliberately does NOT re-derive the caption purely from `pose`/`shot_type`
    buckets -- those are used only as a light fallback prefix if BLIP's
    description doesn't already make the framing obvious, and are never the
    sole content of the caption. This directly avoids the earlier bug where
    every image sharing a pose bucket got the identical caption string.
    """
    try:
        blip_desc = describe_image(pil_image)
    except Exception as e:
        print(f"Warning: BLIP captioning failed ({e}); falling back to a generic description.")
        blip_desc = f"a {class_noun}"

    # BLIP output often starts with "a photo of"/"a woman" etc; strip a
    # leading article+noun so we can splice the trigger in front of it
    # cleanly instead of getting "kmb a woman a woman ...".
    desc = blip_desc
    for prefix in ("a photo of ", "an image of ", "a picture of "):
        if desc.lower().startswith(prefix):
            desc = desc[len(prefix):]
            break

    caption = f"A photo of {trigger}, {desc}"
    if style_hint:
        caption += f", {style_hint}"
    return caption.strip().rstrip(".") + "."
