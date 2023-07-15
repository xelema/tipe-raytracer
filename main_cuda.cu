#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <OpenImageDenoise/oidn.h>

#include "vec3.hu"
#include "ray.hu"
#include "hitinfo.hu"
#include "sphere.hu"
#include "rtutility.hu"
#include "camera.hu"
#include "denoiser.hu"

__host__ __device__ HitInfo hit_sphere(point3 center, double radius, ray r){

    HitInfo hitInfo;
    hitInfo.didHit=false; 

    //si delta>0 alors spherse il y a

    vec3 oc = sub(r.origin, center);
    double a = vec3_dot(r.dir, r.dir);
    double b = 2.0*vec3_dot(oc, r.dir);
    double c = vec3_dot(oc, oc) - radius*radius;
    
    double discriminant = b*b - 4*a*c;

    if (discriminant >= 0){
        double t1 = (-b -sqrt(discriminant))/(2*a);
        double t2 = (-b +sqrt(discriminant))/(2*a);
        
        if (t1 >= 0){
            hitInfo.didHit = true;
            hitInfo.dst = t1;
            hitInfo.hitPoint = ray_at(r, t1);
            hitInfo.normal = vec3_normalize(sub(ray_at(r, t1), center));
        }
        else if (t2 >= 0){
            hitInfo.didHit = true;
            hitInfo.dst = t2;
            hitInfo.hitPoint = ray_at(r, t2);
            hitInfo.normal = vec3_normalize(sub(ray_at(r, t2), center));
        }
    }
    return hitInfo;
}

__host__ __device__ HitInfo closest_hit(ray r, sphere* spheres, int nbSpheres){

    HitInfo closestHit;
    closestHit.didHit=false;
    closestHit.dst=INFINITY; // rien touché pour l'instant

    for(int i=0; i < nbSpheres ; i++){
        sphere s = spheres[i];
        HitInfo hitInfo = hit_sphere(s.center, s.radius, r);

        if (hitInfo.didHit && hitInfo.dst < closestHit.dst){
            closestHit = hitInfo;
            closestHit.mat = s.mat;
        }
    }
    return closestHit;
}

__device__ color tracer(ray r, int nbRebondMax, curandState* globalState, int ind, sphere* spheres, int nbSpheres){

    HitInfo hitInfo = closest_hit(r, spheres, nbSpheres); // cas des lumières
    if (hitInfo.didHit){
        if (hitInfo.mat.emissionStrength > 0){
            color HSL = rgb_to_hsl(hitInfo.mat.emissionColor);
            HSL.e[2] *= 1.20; // luminosité
            HSL.e[1] *= 1.20; // saturation (valeurs subjectives)
            color newCol = hsl_to_rgb(HSL);
            return newCol;
        }
    }
    else return BLACK;
    
    color incomingLight = BLACK;
    color rayColor = WHITE;

    for (int i = 0; i<nbRebondMax; i++){

        HitInfo hitInfo = closest_hit(r, spheres, nbSpheres);

        if (hitInfo.didHit){
            r.origin = hitInfo.hitPoint;
            r.dir = vec3_normalize(add(hitInfo.normal,random_dir_no_norm(globalState, ind))); // sebastian lague

            material mat = hitInfo.mat;
            color emittedLight = multiply_scalar(mat.emissionColor, mat.emissionStrength);

            incomingLight = add(incomingLight,multiply(emittedLight, rayColor));
            rayColor = multiply(mat.diffuseColor, rayColor);
        }
        else{
            break;
        }
    }
    return incomingLight;
}

__global__ void render_canva(color* canva, int largeur_image, int hauteur_image, int nbRayonParPixel, int nbRebondMax, camera cam, curandState* states, sphere* spheres, int nbSpheres) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = j * gridDim.x * blockDim.x + i;

    if (i < largeur_image && j < hauteur_image){
        int pixel_index = j*largeur_image+i;
        color totalLight = BLACK;
        
        for (int k=0; k<nbRayonParPixel; ++k){
            
            double u = ((double)i + randomDouble(states, ind, -0.5, 0.5))/(largeur_image-1);
            double v = ((double)j + randomDouble(states, ind, -0.5, 0.5))/(hauteur_image-1);

            ray r = get_ray(u, v, cam);
            totalLight = add(totalLight, tracer(r, nbRebondMax, states, ind, spheres, nbSpheres));
        }

        canva[pixel_index] = write_color_canva(totalLight, nbRayonParPixel);
    }
}

int main(){

    // CONSTANTES (paramètres de rendu)
    ////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////

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
    int nbRayonParPixel = 2000;
    int nbRebondMax = 5;
    
    int nbThreadsX = 8; // peut dépendre des GPU
    int nbThreadsY = 8; 

    bool useDenoiser = false;

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

    ////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////

    int nbSpheres = sizeof(h_sphere_list) / sizeof(h_sphere_list[0]);

    // temps d'execution
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // nom du fichier
    char nomFichier[100];
    time_t maintenant = time(NULL); // Obtenir l'heure actuelle
    struct tm *temps = localtime(&maintenant); // Convertir en structure tm

    sprintf(nomFichier, "cam_rotation_%dRAYS_%dRB_%02d-%02d_%02dh%02d.ppm", nbRayonParPixel, nbRebondMax-1, temps->tm_mday, temps->tm_mon + 1, temps->tm_hour, temps->tm_min);

    FILE *fichier = fopen(nomFichier, "w");

    // camera
    camera cam = init_camera(origin, target, up, vfov, ratio);

    // tableau pour avoir chaque valeur de pixel au bon endroit (multithread et CUDA du coup)
    color* canva = (color*)malloc((largeur_image * hauteur_image)*sizeof(color));
    for (int i = 0; i < largeur_image*hauteur_image; i++) {
        canva[i] = BLACK;
    }

    // défini la taille des blocks et threads
    dim3 blocks(largeur_image/nbThreadsX+1, hauteur_image/nbThreadsY+1);
    dim3 threads(nbThreadsX, nbThreadsY);

    // alloue la mémoire pour states sur le device (gpu) (necessaire à la fontion de random)
    curandState* states;
    cudaMalloc((void**) &states, (largeur_image * hauteur_image) * sizeof(curandState));

    // alloue la mémoire pour canva sur le device (gpu)
    color* canva_device;
    cudaMalloc((void**)&canva_device, (largeur_image * hauteur_image)*sizeof(color));

    // alloue la mémoire pour d_sphere_list sur le device puis copie h_sphere_list (host) vers le device, optimisation
    sphere* d_sphere_list;
    cudaMalloc((void**)&d_sphere_list, nbSpheres*sizeof(sphere));
    cudaMemcpy(d_sphere_list, h_sphere_list, nbSpheres*sizeof(sphere), cudaMemcpyHostToDevice);

    // initialise les "states" pour la fonction de random
    init_curand_state<<<blocks, threads>>>(states, largeur_image, hauteur_image);

    // lance le rendu de canva
    render_canva<<<blocks, threads>>>(canva_device, largeur_image, hauteur_image, nbRayonParPixel, nbRebondMax, cam, states, d_sphere_list, nbSpheres);

    // copie canva du device (gpu) vers l'host (cpu), puis free la mémoire de canva sur device
    cudaMemcpy(canva, canva_device, (largeur_image * hauteur_image)*sizeof(color), cudaMemcpyDeviceToHost);
    cudaFree(canva_device);

    // utilise le denoiser si l'option est activée
    if (useDenoiser){
        denoiser(largeur_image, hauteur_image, canva, cam, h_sphere_list, nbSpheres);
    }
    
    //base_ppm et canva_to_ppm réecrit ici pour contrer l'appel de fprintf impossible depuis une fonction __host__ __device__
    fprintf(fichier, "P3\n%d %d\n255\n", largeur_image, hauteur_image);

    for (int j = hauteur_image-1; j >= 0  ; j--){ 
        for (int i = 0; i < largeur_image; i++){
            fprintf(fichier, "%d %d %d\n", (int)canva[j*largeur_image+i].e[0], (int)canva[j*largeur_image+i].e[1], (int)canva[j*largeur_image+i].e[2]);
        }
    }
    
    fclose(fichier);
    cudaFree(d_sphere_list);
    cudaFree(states);

    // enregistrer le moment d'arrivée
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    int minutes = (int)(elapsedTime / 60000);
    int seconds = (int)((elapsedTime - minutes * 60000) / 1000);
    
    fprintf(stderr, "\nFini.\n");
    fprintf(stderr, "Temps de rendu : %d min %d sec\n", minutes, seconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check les erreurs CUDA
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    }

	return 0;
}
