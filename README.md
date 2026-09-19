# fireviewer-orchestrator

## Repères documentaires — 19 septembre 2026

- **Rôle :** Coordination stateless de l’exécution des stades sur un état durable possédé par le backend.
- **Statut :** Actif — package v0.1.1.
- **Entrées :** Jobs et contexte backend, disponibilité des providers/composants.
- **Sorties :** Dispatch, retries, receipts et appels aux composants.
- **Limites :** Ne pas créer une deuxième base de vérité, ni dupliquer Part.4, ni ajouter un agent permanent sans besoin démontré.

[Fiche du dépôt](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/repositories/fireviewer-orchestrator.md) · [Architecture](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/ARCHITECTURE.md) · [Statuts et vocabulaire](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/STATUTS_ET_VOCABULAIRE.md).

Cette revue documentaire ne renouvelle aucun test ni aucune réception. Les procédures, versions et preuves techniques ci-dessous conservent leur périmètre et leur date.

> **Source active FV · private.** Exécution, dispatch, reprises et idempotence des agents. Voir [où travailler, quoi commiter et comment reprendre](ORGANISATION.md).

Stateless stage execution and dispatch over durable backend state.

The orchestrator coordinates stage execution through the domain components. The backend remains the authority for durable jobs, leases, permissions, idempotency records, review and publication. Part.4 calculation belongs to `fireviewer-fire-state`; this package does not duplicate that algorithm or become a second incident database.

Python package: `fireviewer_orchestrator`. Version: `0.1.1`.

## Installation

Install the versioned release wheels (including private FireViewer dependencies) from the release bundle. No sibling source checkout is required.

```sh
python -m pip install --find-links /path/to/release/wheels fireviewer-orchestrator==0.1.1
python -m pytest tests -q
```

Optional model/provider environments are separate extras and retain their existing upstream constraints. Model weights, credentials, datasets and local evidence are external inputs.

## Canonical repository and rights

Canonical source: [`fireviewer/fireviewer-orchestrator`](https://github.com/fireviewer/fireviewer-orchestrator). Technical stewardship: FIRE-VIEWER. Repository access: private.

Historical authorship, AGPL-3.0-or-later notices and third-party rights are retained. Technical stewardship and repository placement are not a signed assignment of intellectual-property rights. Any pre-association assets remain subject to their documented licences or agreements.

This repository is the maintained implementation location for the responsibility stated above. Existing schema IDs, algorithm revisions and evidence/publication gates are preserved. Older `firewarning_worker` or backend imports remain compatibility adapters where required; they are not alternative locations for new component logic.

## Delivery and qualification

Versioned packages are distributed through the authorised private release bundles. Current container locks, reconstruction inputs and dated acceptance records are maintained in [fireviewer-docker](https://github.com/fireviewer/fireviewer-docker).

Package installation, CPU/schema tests, service deployment and real-data acceptance are separate checks. CPU/schema tests do not qualify GPU, visual or scientific performance. This documentation update does not publish a package, rebuild an image or change production configuration.

Extraction correspondence and hashes remain in the historical migration dossier. They record the restructuring, not the current deployment state.

## Sources et commandes propres au composant

Commande : `fireviewer-orchestrator` (nécessite le fournisseur runtime configuré ; ne pas la lancer pour vérifier une simple installation). Les exécutions de stades et sessions sont dans `event_pipeline` et `session_runner`. Le backend garde jobs durables, autorisations, idempotence et publication.

Les dépendances de base sont verrouillées avec hashes dans `requirements.lock.txt` (Python 3.13). Installer les wheels privés du même bundle via `--find-links`. Les extras lourds restent liés à leurs versions existantes et ne qualifient aucun GPU. Les commandes de reprise et leurs prérequis sont décrits dans [ORGANISATION.md](ORGANISATION.md). Les reçus du dossier de migration restent des preuves historiques, pas une nouvelle qualification.
