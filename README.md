## Progression
- [x] Étape 0 : Initialisation du projet
- [x] Étape 1 : Extraction du contenu multimodal (texte & images)
- [x] Étape 2 : Embeddings CLIP dans un espace latent commun
- [X] Étape 3 : Indexation dans ChromaDB
- [X] Étape 4 : Recherche multimodale par similarité cosinus
- [ ] Étape 5 : Retriever transformé en outil agentique
- [ ] Étape 6 : Agent ReAct avec LangGraph
- [ ] Étape 7 : Tests et validation


## Structure
- `data/` : PDF source
- `notebooks/` : un notebook par étape, avec les sorties
- `src/embeddings.py` : encodage CLIP (texte et image) et découpage en chunks