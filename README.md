# tipe-raytracer
Implémentation d'un Ray Tracer pour mon exposé de TIPE.

# Paramètres
- Les paramètres de rendu se trouvent dans la fonction main de `main_cuda.cu` :

```C
//format du fichier
double ratio = 4.0 / 3.0;
int largeur_image = 1000;
int hauteur_image = (int)(largeur_image / ratio);

//position de la camera
double vfov = 70; // fov vertical en degrée
    
point3 origin = {{0.34, 0.3, 0.5}}; // position de la camera
point3 target = {{0.0, -0.5, -3}}; // cible de la camera
vec3 up = {{0, 1, 0}}; // permet de modifier la rotation selon l'axe z ({{0, 1, 0}} pour horizontal)

double focus_distance = 3; // distance de mise au point (depth of field)
double ouverture_x = 0.0; // quantité de dof horizontal
double ouverture_y = 0.0; // quantité de dof vertical

//qualité et performance
int nbRayonParPixel = 100;
int nbRebondMax = 5;
    
#define NUM_THREADS 12

bool useDenoiser = true;

bool useAO = true; // occlusion ambiante, rendu environ 2x plus lent
double AO_intensity = 2.5; // supérieur à 1 pour augmenter l'intensité

// chemin des fichiers de mesh
char* obj_file = "model3D/books/book_tri.obj"; // chemin du fichier obj
char* mtl_file = "model3D/books/book_tri.mtl"; // chemin du fichier mtl (textures dans le format PPM P3)

// nom du fichier de sorties
char nomFichier[100];
time_t maintenant = time(NULL); // heure actuelle pour le nom du fichier
struct tm *temps = localtime(&maintenant);
sprintf(nomFichier, "multiple_tex_%dRAYS_%dRB_%02d-%02d_%02dh%02d.ppm", nbRayonParPixel, nbRebondMax-1, temps->tm_mday, temps->tm_mon + 1, temps->tm_hour, temps->tm_min);

//position des sphères dans la scène
sphere sphere_list[10] = {
    //{position du centre x, y, z}, rayon, {couleur de l'objet, couleur d'emission, intensité d'emission (> 1), intensité de reflection (entre 0 et 1)}
    {{{-501,0,0}}, 500, {GREEN, BLACK, 0.0, 0.96}},
    {{{0,-501,0}}, 500, {WHITE, BLACK, 0.0, 0.0}},
    {{{501, 0, 0}}, 500, {RED, BLACK, 0.0, 0.96}},
    {{{-0.5, 1.4, -1.2}}, 0.5, {BLACK, {{1.0, 0.6, 0.2}}, 4.0, 0.0}}, // orange
    {{{0.5, 1.4, -2.2}}, 0.5, {BLACK, {{0.7, 0.2, 1.0}}, 4.0, 0.0}}, // violet
    {{{0.6, -1.4, -1.0}}, 0.5, {BLACK, {{0.55, 0.863, 1.0}}, 2.5, 0.0}}, // bleu clair
    {{{-0.5, -1.4, -3.1}}, 0.5, {BLACK, {{0.431, 1.0, 0.596}}, 2.5, 0.0}}, // vert clair
    {{{0, 0, -504}}, 500, {WHITE, BLACK, 0.0, 0.0}},
    {{{0, 501, 0}}, 500, {WHITE, BLACK, 0.0, 0.0}},
    {{{0.4, -0.5, -3.3}}, 0.5, {SKY, BLACK, 0.0, 0.99}},
    // {{{0.2, -0.7, -2}}, 0.3, {SKY, BLACK, 0.0, 0.5}},
};
```

*******

## Multithreading (CPU)

Compiler avec :
  ```sh
  gcc -o prog main.c -O3 -lm -lpthread -lOpenImageDenoise

  ```
Attention à modifier `#define NUM_THREADS 12` dans `main.c`

## CUDA (GPU)
Si vous avez une carte graphique NVIDIA, \
Compiler avec (en tout cas pour moi) :
  ```sh
  nvcc -o prog_cuda main_cuda.cu -O3 -I"D:\dev\oidn-2.0.1.x64.windows\include" -L"D:\dev\oidn-2.0.1.x64.windows\lib" -lOpenImageDenoise
  ```
Il faut avoir [CUDA](https://developer.nvidia.com/cuda-downloads), et la librairie [OpenImageDenoise](https://github.com/OpenImageDenoise/oidn)\
*<sub>(la version GPU n'est actuellement pas à jour)</sub>*

*******
## Gérer les meshes
- Le mesh doit être trangulaire.  *<sub><sup>(faisable rapidement avec [Blender](https://github.com/blender) par exemple)</sup></sub>*
- Il faut avoir les textures dans un format PPM (P3) et il faut que le fichier .mtl pointe vers la texture en .png.
- Il faut avoir un fichier alpha de la forme "{nom_du_fichier}_alpha.ppm".

Je conseille d'utiliser [ImageMagick](https://github.com/ImageMagick/ImageMagick) et de taper la commande suivante dans un terminal pour avoir les bons fichiers de textures :
```
for file in ./*.png; do
    convert "$file" -alpha extract -colorspace Gray "${file%.png}_alpha.png"
    convert "$file" -compress none "${file%.png}.ppm"
    convert "${file%.png}_alpha.png" -compress none "${file%.png}_alpha.ppm"
done
```

*******

## Différence de temps de rendu
Avec ces paramètres :
```C
double ratio = 4.0 / 3.0;
int largeur_image = 1200;
int hauteur_image = (int)(largeur_image / ratio);

int nbRayonParPixel = 1000;
int nbRebondMax = 6;
    
int nbThreadsX = 8;
int nbThreadsY = 8; 
```
*GTX 1060, Ryzen 5 2600X*
#### 1 THREAD
![rendertime_1thread](https://i.ibb.co/WffpFp7/1-THREADS-render-time-norm.png)


#### 12 THREADS
![rendertime_12thread](https://i.ibb.co/WsfwrTr/12-THREADS-render-time-norm.png)


#### CUDA
![rendertime_CUDA](https://i.ibb.co/v1Mfs5H/CUDA-render-time-norm.png)



*******

# Options
## Denoiser (Intel Open Image Denoise)

<p>
    <img src="https://github.com/xelema/tipe-raytracer/blob/0c561730cfbd47d421efb32ebc1d42bd2efc005f/results/without_denoiser2_2000RAYS_5RB_13-07_00h24.png" width="500", alt>
    <br>
    <em>SANS</em>
</p>
<br>
<p>
    <img src="https://github.com/xelema/tipe-raytracer/blob/0c561730cfbd47d421efb32ebc1d42bd2efc005f/results/with_denoiser2_2000RAYS_5RB_13-07_00h21.png" width="500">
    <br>
    <em>AVEC</em>
</p>

## Occlusion Ambiante

Avec ce paramètre : 
```C
double AO_intensity = 8.0;
```

<p>
    <img src="https://github.com/xelema/tipe-raytracer/blob/cdfc8376f58230350e8e748d53e0b9c83f27951a/results/withoutAO_10000RAYS_4RB_15-07_16h31.png" width="500">
    <br>
    <em>SANS</em>
</p>
<br>
<p>
    <img src="https://github.com/xelema/tipe-raytracer/blob/cdfc8376f58230350e8e748d53e0b9c83f27951a/results/withAO_10000RAYS_4RB_15-07_15h58.png" width="500">
    <br>
    <em>AVEC</em>
</p>

