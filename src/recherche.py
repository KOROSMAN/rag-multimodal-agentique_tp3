"""Recherche multimodale (texte ou image) par similarité cosinus."""
from src.embeddings import encoder_textes, encoder_images
from src.indexation import ouvrir_collection, ROOT

collection = ouvrir_collection()


def rechercher(texte=None, image=None, type_resultat=None, k=5):
    """
    Requête = un texte OU un chemin d'image.
    type_resultat : None (tout), "texte" ou "image".
    Renvoie une liste de dicts triés du plus similaire au moins similaire.
    """
    if (texte is None) == (image is None):
        raise ValueError("Fournis soit un texte, soit une image (pas les deux).")
    if type_resultat not in (None, "texte", "image"):
        raise ValueError("type_resultat doit être None, 'texte' ou 'image'.")

    # 1. Encoder la requête dans l'espace CLIP
    if texte is not None:
        vecteur = encoder_textes([texte])[0]
    else:
        vecteur = encoder_images([image])[0]

    # 2. Chercher les k plus proches voisins
    resultats = collection.query(
        query_embeddings=[vecteur.tolist()],
        n_results=k,
        where={"type": type_resultat} if type_resultat else None,
        include=["documents", "metadatas", "distances"],
    )

    # 3. Remettre en forme (Chroma renvoie des listes de listes : une par requête)
    return [
        {
            "id": id_,
            "type": meta["type"],
            "page": meta["page"],
            "contenu": doc,
            "chemin": meta.get("chemin"),          # None pour les textes
            "similarite": round(1 - distance, 3),  # distance cosinus -> similarité
        }
        for id_, doc, meta, distance in zip(
            resultats["ids"][0], resultats["documents"][0],
            resultats["metadatas"][0], resultats["distances"][0],
        )
    ]