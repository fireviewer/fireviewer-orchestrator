# Planification par preuve

La version candidate 0.1.2 fournit `plan_incident_enrichment(TemporalEvidence)` et adopte les contrats
0.1.3. Cette fonction pure décrit les besoins : vision selon le média, transcription pour l'audio,
géolocalisation, supervision et fusion après admission. Une date inconnue bloque la reconstruction ;
un retrait demande un rejeu sans exécuter de modèle.

Ce planificateur ne constitue pas un nouveau moteur autonome de dispatch. L'exécution utilise encore
le pipeline conditionnel existant. Le backend réserve la preuve à l'enqueue/claim et ajoute les versions
après validation des reçus. Il possède les files durables, reprises, droits et décisions humaines.
La fin d'un worker ne publie aucun état.

Les dépendances d'ingestion, vision, géolocalisation et supervision passent en 0.1.2 pour porter les
nouveaux contrats sans conflits de métadonnées. Leurs algorithmes et modèles ne sont pas modifiés.
Les wheels conservés dans `vendor/` sont vérifiés par SHA-256 et licences, sans repaqueter une ancienne
release sous le même numéro. `python tools/ci.py verify` teste l'installation isolée et `uv pip check`.

Les services/modèles réels, les fournisseurs et le GPU gardent leurs propres critères de réception.
