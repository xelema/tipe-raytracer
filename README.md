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
    vec3 up = {{0, 1, 0}}; // permet de modifier la rotation selon l'axe z ({{0, 1, 0}} pour horizontal)

    //qualité et performance
    int nbRayonParPixel = 2000;
    int nbRebondMax = 5;
    
    int nbThreadsX = 8; // peut dépendre des GPU
    int nbThreadsY = 8; 

    bool useDenoiser = true; // utilise ou non le denoiser

    //position des sphères dans la scène
    sphere h_sphere_list[10] = {
        //{position du centre x, y, z}, rayon, {couleur de l'objet, couleur d'emission, force d'emission}
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

- Le nom du fichier se trouve dans `main_cuda.cu` :

```C
    sprintf(nomFichier, "CUDA_%dRAYS_%dRB_%02d-%02d_%02dh%02d.ppm", nbRayonParPixel, nbRebondMax-1, temps->tm_mday, temps->tm_mon + 1, temps->tm_hour, temps->tm_min);
```

# CUDA

- Compiler avec (en tout cas pour moi)
  ```sh
  nvcc -o prog_cuda main_cuda.cu -O3 -I"D:\dev\oidn-2.0.1.x64.windows\include" -L"D:\dev\oidn-2.0.1.x64.windows\lib" -lOpenImageDenoise
  ```
  il faut avoir nvcc (CUDA), et la librairie OpenImageDenoise (https://github.com/OpenImageDenoise/oidn)

# Temps de rendu
#### 1 THREAD
![rendertime_1thread](https://i.ibb.co/WffpFp7/1-THREADS-render-time-norm.png)


#### 12 THREADS
![rendertime_12thread](https://i.ibb.co/WsfwrTr/12-THREADS-render-time-norm.png)


#### CUDA
![rendertime_CUDA](https://i.ibb.co/v1Mfs5H/CUDA-render-time-norm.png)


####
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

# Denoiser (Intel Open Image Denoise)

### SANS
![without](https://github.com/xelema/tipe-raytracer/blob/0c561730cfbd47d421efb32ebc1d42bd2efc005f/results/without_denoiser2_2000RAYS_5RB_13-07_00h24.png)

### AVEC
![with](https://github.com/xelema/tipe-raytracer/blob/0c561730cfbd47d421efb32ebc1d42bd2efc005f/results/with_denoiser2_2000RAYS_5RB_13-07_00h21.png)

