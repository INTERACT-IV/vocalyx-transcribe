# Analyse de faisabilité : Prompting système dans la brique transcription

## Résumé exécutif

**✅ FAISABLE** - L'implémentation du prompting système (via `initial_prompt`) est techniquement faisable et **ne nécessite PAS de recharger le modèle** entre les transcriptions.

## Contexte technique

### Support dans faster-whisper et WhisperX

Le paramètre `initial_prompt` est nativement supporté par :

1. **`faster-whisper`** (utilisé directement par votre brique transcription) ✅
2. **`WhisperX`** (qui utilise `faster-whisper` sous le capot) ✅

```python
segments, info = self.model.transcribe(
    str(file_path),
    language=self.config.language or None,
    task="transcribe",
    initial_prompt="Votre prompt système ici",  # ✅ Supporté dans les deux
    # ... autres paramètres
)
```

**Note** : WhisperX propage le `initial_prompt` depuis la ligne de commande jusqu'à `faster-whisper`, confirmant que c'est une fonctionnalité standard et bien supportée.

### Différence avec un "system prompt" classique

⚠️ **Important** : Le `initial_prompt` dans Whisper/faster-whisper n'est **pas** un "system prompt" au sens des LLMs modernes (comme GPT). C'est plutôt :

- Un **prompt initial** qui guide la première fenêtre de transcription (~30 secondes)
- Un **contexte textuel** fourni au début de la transcription pour améliorer la précision
- Un **guide de style/terminologie** pour le modèle

## Architecture actuelle

### Structure du code

```
vocalyx-transcribe/
├── transcription_service.py    # Service principal (TranscriptionService)
├── worker.py                   # Worker Celery (get_transcription_service)
└── infrastructure/models/
    └── model_cache.py          # Cache LRU des modèles
```

### Flux actuel

1. **Chargement du modèle** (lazy loading) :
   - `TranscriptionService._load_model()` charge `WhisperModel` une seule fois
   - Le modèle est mis en cache via `ModelCache` (LRU, max 10 modèles)

2. **Transcription** :
   - `transcribe_segment()` appelle `self.model.transcribe()`
   - Le modèle est réutilisé pour toutes les transcriptions

3. **Cache de modèles** :
   - Un modèle par nom (tiny, base, small, medium, large-v3, etc.)
   - Pas de distinction par prompt actuellement

## Faisabilité technique

### ✅ Avantages

1. **Pas de rechargement nécessaire** :
   - Le `initial_prompt` est passé comme paramètre à chaque appel
   - Le modèle reste en mémoire, seul le prompt change
   - **Performance optimale** : pas de latence de chargement

2. **Intégration simple** :
   - Ajout d'un paramètre optionnel `initial_prompt` dans les méthodes existantes
   - Propagation via la chaîne d'appels (worker → service → model.transcribe)
   - Compatible avec le cache de modèles existant

3. **Flexibilité** :
   - Prompt différent par transcription possible
   - Pas de limitation sur le nombre de prompts différents
   - Peut être configuré au niveau de la transcription individuelle

### ⚠️ Limitations et considérations

1. **Portée du prompt** :
   - Le `initial_prompt` influence principalement la **première fenêtre** (~30s)
   - Pour les fichiers longs, l'effet diminue après le début
   - Pas de "system prompt" global pour tout le fichier

2. **Gestion du cache** :
   - **Option A** : Un modèle par combinaison (modèle + prompt) → plus de mémoire
   - **Option B** : Un modèle par type, prompt passé dynamiquement → **recommandé**
   - Le cache actuel (Option B) est déjà optimal

3. **Performance** :
   - Pas d'impact sur la vitesse de transcription
   - Légère augmentation mémoire si on tokenise le prompt (négligeable)

## Implémentation proposée

### Niveau 1 : Support basique (recommandé)

```python
# transcription_service.py
def transcribe_segment(
    self, 
    file_path: Path, 
    use_vad: bool = True, 
    retry_without_vad: bool = True,
    initial_prompt: Optional[str] = None  # ✅ Nouveau paramètre
) -> Tuple[str, List[Dict], str]:
    # ...
    segments, info = self.model.transcribe(
        str(file_path),
        language=self.config.language or None,
        task="transcribe",
        initial_prompt=initial_prompt,  # ✅ Passé au modèle
        # ... autres paramètres
    )
```

**Propagation** :
- `transcribe()` → `transcribe_segment()` → `model.transcribe()`
- `worker.py` → récupère depuis la transcription API → passe au service

### Niveau 2 : Support avancé (optionnel)

1. **Prompts prédéfinis** :
   - Dictionnaire de prompts par domaine (médical, juridique, technique)
   - Sélection automatique selon métadonnées

2. **Prompts dynamiques** :
   - Génération de prompt basé sur le contexte (langue, domaine détecté)
   - Intégration avec l'enrichissement pour prompts adaptatifs

3. **Validation** :
   - Limite de longueur du prompt (max ~448 tokens)
   - Vérification de la cohérence avec la langue détectée

## Impact sur l'architecture

### Modifications nécessaires

1. **transcription_service.py** :
   - Ajouter `initial_prompt` paramètre dans `transcribe_segment()` et `transcribe()`
   - Propagation vers `model.transcribe()`

2. **worker.py** :
   - Récupérer `initial_prompt` depuis les métadonnées de transcription
   - Passer au service de transcription

3. **API** (vocalyx-api) :
   - Ajouter champ `initial_prompt` dans le modèle de transcription
   - Validation et stockage en base

### Pas de modifications nécessaires

- ✅ `ModelCache` : fonctionne tel quel (pas de distinction par prompt)
- ✅ Cache Redis : pas d'impact
- ✅ Mode distribué : compatible (prompt passé dans métadonnées)

## Comparaison avec WhisperX

### ✅ WhisperX supporte aussi `initial_prompt`

**Confirmation** : WhisperX supporte bien le `initial_prompt` ! 

Dans le code source de WhisperX (`whisperx/transcribe.py` ligne 108 et `whisperx/asr.py` lignes 60-63), on peut voir :

```python
# Dans transcribe.py
asr_options = {
    "initial_prompt": args.pop("initial_prompt"),  # ✅ Supporté
    # ... autres options
}

# Dans asr.py - generate_segment_batched()
if options.initial_prompt is not None:
    initial_prompt = " " + options.initial_prompt.strip()
    initial_prompt_tokens = tokenizer.encode(initial_prompt)
    all_tokens.extend(initial_prompt_tokens)
```

### WhisperX utilise faster-whisper sous le capot

WhisperX utilise `faster-whisper` sous le capot, donc le comportement est identique :

- ✅ Support de `initial_prompt` (confirmé dans le code source)
- ✅ Pas de rechargement nécessaire
- ⚠️ Même limitation : prompt principalement pour la première fenêtre (~30s)

### Différence principale

WhisperX ajoute des couches supplémentaires (VAD, alignment, diarization) mais le prompting fonctionne exactement de la même manière que dans faster-whisper pur.

**Conclusion** : Votre brique transcription (qui utilise directement `faster-whisper`) peut implémenter le `initial_prompt` exactement comme WhisperX le fait.

## Recommandations

### ✅ Implémentation recommandée

1. **Phase 1** : Support basique du `initial_prompt`
   - Ajout du paramètre dans les méthodes existantes
   - Propagation depuis l'API jusqu'au modèle
   - Tests avec différents prompts

2. **Phase 2** (optionnel) : Améliorations
   - Prompts prédéfinis par domaine
   - Validation et limites
   - Monitoring de l'impact sur la qualité

### ⚠️ Points d'attention

1. **Longueur du prompt** :
   - Limite technique : ~448 tokens (max_length du modèle)
   - Recommandation : limiter à ~200 tokens pour sécurité

2. **Langue du prompt** :
   - Le prompt doit être dans la même langue que l'audio
   - Validation recommandée côté API

3. **Performance** :
   - Pas d'impact mesurable attendu
   - Monitoring recommandé pour confirmer

## Conclusion

**Faisabilité : ✅ TRÈS FAISABLE**

- ✅ Support natif dans faster-whisper
- ✅ Pas de rechargement de modèle nécessaire
- ✅ Intégration simple dans l'architecture existante
- ✅ Compatible avec le cache de modèles actuel
- ✅ Pas d'impact négatif sur les performances

**Effort estimé** : 2-4 heures pour l'implémentation basique

**Risques** : Faibles (fonctionnalité optionnelle, pas de breaking changes)

---

## ✅ Implémentation réalisée

### Modifications apportées

1. **`transcription_service.py`** :
   - Ajout du paramètre `initial_prompt: Optional[str] = None` dans `transcribe_segment()` et `transcribe()`
   - Passage conditionnel de `initial_prompt` à `model.transcribe()` (comme WhisperX)
   - Dans `_transcribe_sequential()`, utilisation uniquement pour le premier segment (comportement identique à WhisperX)
   - Logging du prompt utilisé pour le debugging

2. **`worker.py`** :
   - Récupération de `initial_prompt` depuis les métadonnées de transcription (`transcription.get('initial_prompt')`)
   - Passage au service de transcription uniquement en mode classique (`_transcribe_classic_mode()`)
   - **Mode distribué** : `initial_prompt` n'est PAS utilisé (comme demandé)
   - Commentaire explicite dans `transcribe_segment_task()` indiquant que le prompt n'est pas utilisé en mode distribué

### Comportement (identique à WhisperX)

- ✅ `initial_prompt` guide uniquement la première fenêtre de transcription (~30 secondes)
- ✅ Pour les fichiers avec plusieurs segments, le prompt n'est utilisé que pour le premier segment
- ✅ Le prompt est encodé automatiquement par `faster-whisper` (comme dans WhisperX)
- ✅ Pas de rechargement de modèle nécessaire
- ✅ Compatible avec le cache de modèles existant

### Utilisation

Pour utiliser `initial_prompt`, il suffit d'ajouter le champ `initial_prompt` dans les métadonnées de transcription côté API :

```python
transcription = {
    "file_path": "...",
    "vad_enabled": True,
    "whisper_model": "large-v3",
    "initial_prompt": "Ceci est une transcription d'une réunion technique sur l'architecture logicielle."  # ✅ Nouveau
}
```

Le prompt sera automatiquement utilisé uniquement en mode classique (non distribué).
