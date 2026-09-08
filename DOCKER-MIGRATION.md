# Docker — phase du 8 septembre 2026

La composition isolée, les références d'images et les contextes de reconstruction verrouillés sont dans le dépôt privé `fireviewer/fireviewer-docker`, release `container-20260908-rc1`.
Les recettes propres à ce composant sont dans `docker/`. Les anciens Dockerfiles, publications et consommateurs déployés restent conservés. Les wheels `v0.1.0` ne sont pas remplacés.
Le Map Builder reste chez UWD. Son image Linux est une candidate de qualification : la répétition Linux est exacte, la parité binaire avec Windows n'est pas acceptée.
Les services GPU, poids de modèles et connexions aux fournisseurs gardent leur qualification distincte. Aucun changement de visibilité ni acte juridique de cession ne découle de cette préparation.
