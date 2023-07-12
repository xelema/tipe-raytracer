# tipe-raytracer
Implémentation d'un Ray Tracer pour mon exposé de TIPE.

# Paramètres
- Les paramètres de rendu se trouvent dans la fonction main de `main_cuda.cu` :

```C
    double ratio = 4.0 / 3.0;
    int largeur_image = 1200;
    int hauteur_image = (int)(largeur_image / ratio);

    int nbRayonParPixel = 1000;
    int nbRebondMax = 6;
    
    int nbThreadsX = 8; // peut dépendre des GPU
    int nbThreadsY = 8; 
```

- Les positions des sphères dans `main_cuda.cu` :

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

- Le nom du fichier se trouve dans `main_cuda.cu` :

```C
    sprintf(nomFichier, "CUDA_%dRAYS_%dRB_%02d-%02d_%02dh%02d.ppm", nbRayonParPixel, nbRebondMax-1, temps->tm_mday, temps->tm_mon + 1, temps->tm_hour, temps->tm_min);
```

# CUDA

- Compiler avec
  ```sh
  nvcc -o prog main_cuda.cu -O3
  ```

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
![image_rendered](https://github.com/xelema/tipe-raytracer/blob/a124fd085db0d1ac1ac3c800215fecaf7c41918e/results/CUDA_1000RAYS_5RB_11-07_15h36.png)

# Denoiser (Intel Open Image Denoise)

### SANS
![without](https://github.com/xelema/tipe-raytracer/blob/ece72fb18e277437a994bfc02aebaf22de634972/results/without_denoise_5000RAYS_5RB_12-07_23h43.png)

### AVEC
![with](https://github.com/xelema/tipe-raytracer/blob/ece72fb18e277437a994bfc02aebaf22de634972/results/with_denoise_5000RAYS_5RB_12-07_23h34.png)

