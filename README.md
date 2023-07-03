# tipe-raytracer
Implémentation d'un Ray Tracer pour mon exposé de TIPE.

# Paramètres
- Les paramètres de rendu se trouvent dans `rtutility.h` :

```C
const double ratio = 16.0/10.0;

const int largeur_image = 1000;
const int hauteur_image = (int)(largeur_image / ratio);

const int nbRayonParPixel = 100;
const int nbRebondMax = 5;
```

- Les positions des sphères dans `main.c` :

```C
const sphere sphere_list[10] = {
    {{{-501,0,0}}, 500, {GREEN, BLACK, 0.0}},                 
    {{{0,-501,0}}, 500, {WHITE, BLACK, 0.0}},                 
    {{{501, 0, 0}}, 500, {RED, BLACK, 0.0}},                  
    {{{-0.5, 1.4, -3}}, 0.5, {BLACK, {{1.0, 0.6, 0.2}}, 8.0}},   
    {{{0.5, 1.4, -3}}, 0.5, {BLACK, {{0.7, 0.2, 1.0}}, 8.0}},   
    {{{-0.5, -1.4, -3}}, 0.5, {BLACK, {{0.55, 0.863, 1.0}}, 5.0}},   
    {{{0.5, -1.4, -3}}, 0.5, {BLACK, {{0.431, 1.0, 0.596}}, 5.0}},   
    {{{0, 0, -504}}, 500, {WHITE, BLACK, 0.0}},               
    {{{0, 501, 0}}, 500, {WHITE, BLACK, 0.0}},                
    {{{0, 0, -3}}, 0.5, {SKY, BLACK, 0.0}}                    
    };
```

# Rendu
- Il suffit de compiler le programme puis de l'éxecuter, le nom du fichier se trouve dans `main.c` :

```C
sprintf(nomFichier, "newscene_lambertslaw_%dRAYS_%dRB_%02d-%02d_%02dh%02d.ppm", nbRayonParPixel, nbRebondMax-1, temps->tm_mday, temps->tm_mon + 1, temps->tm_hour, temps->tm_min);
```



