# 🐾 EverMate.AI (v1.0 bientôt)
Votre compagnon IA local · Priorité à la vie privée · Hors ligne autant que possible

[English (Canada)](README.en-CA.md) · [中文](README.zh-CN.md) · [Português (Brasil)](README.pt-BR.md) · [Français (Canada)](README.fr-CA.md) · [日本語](README.ja-JP.md)

## ✨ En bref
EverMate.AI rend les **échanges d’accompagnement à long terme** vraiment utilisables et maîtrisés : exécution locale par défaut, pas d’envoi, et deux atouts maison — **Mémoire en trois niveaux** + **Index local extensible**.

## 🍱 Nos atouts
### 1) Mémoire en trois niveaux (Core → Persona → Vault)
- **Core** : thèmes fréquents + indices de style/réponse (ex. ton chaleureux, exemples). La plupart des demandes se règlent ici — vite et bien.  
- **Persona** : préférences de communication, densité d’info, intérêts durables, éléments « à éviter » — ≤8 points. LLM local via Ollama si dispo, sinon heuristiques.  
- **Vault** : le reste est **découpé et stocké sur disque**, puis extrait à la demande — pas de pavé, seulement la **bonne phrase** au bon moment.  
- **Conflit** : si l’historique contredit votre **entrée actuelle**, c’est l’actuelle qui prime.

### 2) Index local extensible (Grand · Local · Rapide)
- **Découpage** : ~2,8 K caractères par bloc (ajustable), pensé pour des millions de mots.  
- **Index inversé** : SQLite (`terms/postings/chunks`) en mode WAL pour des E/S stables.  
- **BM25** : classe les blocs et remonte la **meilleure phrase source** comme preuve.  
- **Construction en flux & mises à jour incrémentales** : glisser-déposer `.docx/.txt` pour indexer en lecture; les nouvelles conversations s’ajoutent tour par tour, Core/Persona se rafraîchissent périodiquement.

## 🚀 Démarrage rapide
1. `python app.py` à la racine du projet.  
2. Deux parcours :  
   - **Importer un historique** : glisser-déposer `.docx/.txt`, cliquer « Build/Rebuild Memory », puis discuter.  
   - **Nouveau compagnon** : lancez la discussion. Chaque tour s’ajoute à l’index; Core/Persona se mettent à jour au fil de l’eau.  
3. Optionnel : configurer **Ollama** (`OLLAMA_URL`, `OLLAMA_MODEL`) pour affiner la Persona en local.

## 💼 Principe (en une phrase)
D’abord **Core** (style & récurrences), puis **Persona** (vos préférences), et **Vault** pour les citations. Contexte **court et pertinent**.

## 🧲 Import vs. Nouveau
- **Glisser-déposer** : `.docx` via bibliothèque standard ; `.txt` en streaming.  
- **Nouveau** : `append_turn` écrit en incrémental ; rafraîchissement par défaut toutes **20** nouveautés.

## 🔧 Réglages
`CHUNK_CHARS`=2800 · `CORE_TOP_TERMS`=50 · `PERSONA_MAX_BULLETS`=8 · `REFRESH_EVERY`=20 · `retrieve(query,k)`=4–8

## 🛡️ Vie privée & priorité au local
Tout reste en local. En cas de modèle distant, vérifiez vos exigences légales et réseau. Chiffrez/sauvegardez `memory`.

## ❓FAQ
Persona inchangée ? Seuil de rafraîchissement non atteint — continuez ou reconstruisez; baissez `REFRESH_EVERY` au besoin.  
Voir la **phrase d’origine** ? Oui — Vault renvoie la phrase/section la plus pertinente.  
Index trop volumineux ? Augmentez `CHUNK_CHARS`, archivez d’anciens `chunks`, ou séparez `memory` par compagnon.  
Sans LLM local ? Seule la Persona perd en finesse; le flux reste stable.  
Sensation d’être « figé » par l’historique ? L’entrée **courante** l’emporte; ajustez `01_core.md` / `02_persona.md` puis reconstruisez.

## 🗺️ Feuille de route
Recherche hybride (BM25 + embeddings locaux), thèmes & frise, panneau d’explicabilité, décote mémoire, fiches d’événements.

## 📦 Arborescence
```
memory/
  index.sqlite
  chunks/
  uploads/
  buffer.txt
  01_core.md
  02_persona.md
  03_vault.md
```

## 📜 Licence
MIT (voir LICENSE)
