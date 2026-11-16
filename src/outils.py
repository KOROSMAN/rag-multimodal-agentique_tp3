"""Outil de recherche documentaire multimodale, utilisable par un agent LLM.

Principe MIND / BODY : cet outil est le BODY (il récupère des sources),
c'est le LLM de l'agent (MIND) qui rédige la réponse. L'outil ne génère rien.
"""
import re
from functools import lru_cache
from typing import Literal

from langchain_core.tools import tool

from src.indexation import ROOT
from src.recherche import rechercher, rechercher_hybride

TEXTES_DIR = ROOT / "resultats_pdf" / "textes"


@lru_cache(maxsize=None)
def legendes_page(page):
    """Légendes "Figure N: ..." trouvées dans le texte extrait d'une page.

    Les images n'ont pas de texte associé dans la base : on donne au LLM la légende
    de la page pour qu'il sache de quelle figure il s'agit.
    """
    fichier = TEXTES_DIR / f"page_{page:03d}.txt"
    if not fichier.exists():
        return []
    texte = fichier.read_text(encoding="utf-8")
    # Une légende va de "Figure N:" jusqu'à la prochaine ligne vide
    legendes = re.findall(r"Figure \d+:.*?(?=\n\s*\n|\Z)", texte, flags=re.S)
    return [" ".join(l.split())[:200] for l in legendes]


@tool(response_format="content_and_artifact")
def retriever_tool(
    requete: str,
    type_resultat: Literal["tous", "texte", "image"] = "tous",
    k: int = 5,
):
    """Recherche des passages et des figures dans l'article "Attention Is All You Need" (le Transformer).

    Utilise cet outil pour TOUTE question sur le contenu de l'article : architecture
    encodeur/décodeur, self-attention, scaled dot-product attention, multi-head attention,
    positional encoding, entraînement, hyperparamètres, résultats (BLEU), ou ses figures.
    Ne réponds pas de mémoire : cherche d'abord. L'outil renvoie des sources numérotées
    [S1], [S2]... avec leur page ; il ne rédige pas de réponse.

    Args:
        requete: requête de recherche courte et précise, rédigée EN ANGLAIS (l'article est
            en anglais), avec les mots-clés de l'article. Ex : "scaled dot-product attention
            scaling factor sqrt d_k".
        type_resultat: "texte" pour des passages explicatifs, "image" pour des figures ou
            schémas, "tous" (par défaut) pour des passages ET les figures les plus proches.
        k: nombre de passages texte à récupérer (entre 1 et 10). Garde 5 par défaut :
            moins de 3 passages donne souvent un contexte insuffisant.
    """
    k = max(1, min(int(k), 10))

    if type_resultat == "tous":
        resultats = rechercher_hybride(requete, k_texte=k, k_image=2)
    else:
        resultats = rechercher(texte=requete, type_resultat=type_resultat, k=k)

    if not resultats:
        return "Aucun résultat trouvé dans l'article.", []

    lignes = []
    for i, r in enumerate(resultats, start=1):
        if r["type"] == "texte":
            lignes.append(f"[S{i}] (texte, page {r['page']}) {r['contenu']}")
        else:
            legende = " | ".join(legendes_page(r["page"])) or "légende non trouvée"
            lignes.append(f"[S{i}] (image, page {r['page']}) Figure de l'article. "
                          f"Légende sur cette page : {legende}")

    contenu = "\n\n".join(lignes)
    return contenu, resultats          # (ce que voit le LLM, données brutes pour le programme)
