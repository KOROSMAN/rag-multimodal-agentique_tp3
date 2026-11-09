"""Encodage CLIP du texte et des images dans un espace vectoriel commun."""
import torch
import open_clip
from PIL import Image
from open_clip.tokenizer import SimpleTokenizer

# --- Chargement du modèle (fait une seule fois, au moment de l'import) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
bpe = SimpleTokenizer()


@torch.inference_mode()
def encoder_textes(textes):
    """Liste de textes -> tenseur [N, 512] de vecteurs normalisés."""
    tokens = tokenizer(textes).to(DEVICE)
    v = model.encode_text(tokens).float()
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v.cpu()


@torch.inference_mode()
def encoder_images(chemins):
    """Liste de chemins d'images -> tenseur [N, 512] de vecteurs normalisés."""
    tenseurs = []
    for c in chemins:
        with Image.open(c) as img:
            tenseurs.append(preprocess(img.convert("RGB")))
    lot = torch.stack(tenseurs).to(DEVICE)
    v = model.encode_image(lot).float()
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v.cpu()


def decouper_texte(texte, taille=60, chevauchement=10):
    """Découpe un texte en chunks de `taille` tokens BPE avec chevauchement."""
    if not 0 <= chevauchement < taille:
        raise ValueError("Le chevauchement doit être inférieur à la taille.")
    tokens = bpe.encode(texte)
    pas = taille - chevauchement
    for debut in range(0, len(tokens), pas):
        fragment = bpe.decode(tokens[debut:debut + taille]).strip()
        if fragment:
            yield fragment
        if debut + taille >= len(tokens):
            break