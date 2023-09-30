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
    double vfov = 110; // fov vertical en degrée
    point3 origin = {{-0.9, 0.9, -3.8}}; // position de la camera
    point3 target = {{0.2, 0, -2.8}}; // cible de la camera
    vec3 up = {{0, 1, 0.2}}; // permet de modifier la rotation selon l'axe z ({{0, 1, 0}} pour horizontal)

    //qualité et performance
    int nbRayonParPixel = 100;
    int nbRebondMax = 5;
    
    int nbThreadsX = 8; // peut dépendre des GPU
    int nbThreadsY = 8; 

    bool useDenoiser = true;

    bool useAO = false; // occlusion ambiante, rendu environ 2x plus lent
    double AO_intensity = 5.0; // supérieur à 1 pour augmenter l'intensité

    //position des sphères dans la scène
    sphere h_sphere_list[10] = {
        //{position du centre x, y, z}, rayon, {couleur de l'objet, couleur d'emission, force d'emission}
        {{{-501,0,0}}, 500, {GREEN, BLACK, 0.0}},                 
        {{{0,-501,0}}, 500, {WHITE, BLACK, 0.0}},                 
        {{{501, 0, 0}}, 500, {RED, BLACK, 0.0}},                  
        {{{-0.5, 1.4, -3}}, 0.5, {BLACK, {{1.0, 0.6, 0.2}}, 4}},   
        {{{0.5, 1.4, -3}}, 0.5, {BLACK, {{0.7, 0.2, 1.0}}, 4}},   
        {{{-0.5, -1.4, -3}}, 0.5, {BLACK, {{0.55, 0.863, 1.0}}, 2.5}},   
        {{{0.5, -1.4, -3}}, 0.5, {BLACK, {{0.431, 1.0, 0.596}}, 2.5}},   
        {{{0, 0, -504}}, 500, {WHITE, BLACK, 0.0}},               
        {{{0, 501, 0}}, 500, {WHITE, BLACK, 0.0}},                
        {{{0, 0, -3}}, 0.5, {SKY, BLACK, 0.0}}                    
    };

    // nom du fichier
    char nomFichier[100];
    time_t maintenant = time(NULL); // Obtenir l'heure actuelle
    struct tm *temps = localtime(&maintenant); // Convertir en structure tm

    sprintf(nomFichier, "AO_%dRAYS_%dRB_%02d-%02d_%02dh%02d.ppm", nbRayonParPixel, nbRebondMax-1, temps->tm_mday, temps->tm_mon + 1, temps->tm_hour, temps->tm_min);
```

*******

## Multithreading

- Compiler avec 
  ```sh
  gcc -o prog main.c -O3 -lm -lpthread -lOpenImageDenoise

  ```
Attention à bien modifier `#define NUM_THREADS 12` dans `main.c`

## CUDA
Si vous avez une carte graphique NVIDIA :
- Compiler avec (en tout cas pour moi)
  ```sh
  nvcc -o prog_cuda main_cuda.cu -O3 -I"D:\dev\oidn-2.0.1.x64.windows\include" -L"D:\dev\oidn-2.0.1.x64.windows\lib" -lOpenImageDenoise
  ```
Il faut avoir [CUDA](https://developer.nvidia.com/cuda-downloads), et la librairie [OpenImageDenoise](https://github.com/OpenImageDenoise/oidn)

*******

# Temps de rendu
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

