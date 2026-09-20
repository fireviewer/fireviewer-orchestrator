# Migration du juge Bonsaï — 21 septembre 2026

Le juge `consensus_judge` utilise Bonsaï 2 27B PTQ1_0, avec projecteur visuel Q8,
révision `6ed5e12bf84b7a63069882c91dd9e9218647d17b`. La recherche reste Qwen3-14B.
Le profil courant est L4 24 Go, **expérimental** jusqu'au comparatif complet ; le
manifeste A40 historique reste lisible pour audit. Changer le code de `main` ne
qualifie pas le déploiement ni la précision du modèle.

L'adaptateur lit les preuves disponibles, sélectionne un candidat existant ou
s'abstient. Il conserve les contrôles humains ; pas de publication autonome, pas
de coordonnées inventées, pas de fallback CPU silencieux. Les fenêtres au-delà
de huit images et les cas audio sans support provoquent une abstention.

Deux sondes GPU ont fonctionné (rapport, photo) avec environ 8 GiB échantillonnés
pour le juge. L'ensemble du pipeline demande davantage ; Qwen3.5 utilise environ
18,25 GiB sur un lot partiel. Le comparatif de journées complètes reste inachevé.

Les réparations associées portent sur l'autocast D-FINE, Florence natif sous
Transformers 5.14.1 et la normalisation d'une seule enveloppe JSON Qwen.
D-FINE standard COCO est le substitut explicite disponible et n'a pas de classe
feu/fumée ; le nom de constante historique reste pour compatibilité.
Les diagnostics d'erreur ne journalisent plus le message ou la réponse source.

Validation ciblée combinée : 70 tests passent dans l'image GPU r5 sur CPU.
Les nouvelles sources ne constituent pas encore une nouvelle image déployée.

Les expérimentations Jev restent indépendantes et textuelles :
https://github.com/fireviewer/fireviewer-jev-experiments
Leur protocole compare une seule brique à la fois avec observations visuelles conservées.
