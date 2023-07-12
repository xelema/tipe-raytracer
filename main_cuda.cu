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


__host__ __device__ HitInfo hit_sphere(point3 center, double radius, ray r){

    HitInfo hitInfo;
    hitInfo.didHit=false; 

    //si delta>0 alors sphere il y a

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

__host__ __device__ HitInfo closest_hit(ray r){


    sphere sphere_list[7] = {
    //MURS//
    {{{-501,0,0}}, 500, {GREEN, BLACK, 0.0}},                 
    // mur gauche vert
    {{{0,-501,0}}, 500, {WHITE, BLACK, 0.0}},                 
    // sol blanc
    {{{501, 0, 0}}, 500, {RED, BLACK, 0.0}},                  
    // mur droite rouge
    {{{0, 0, -504}}, 500, {WHITE, BLACK, 0.0}},               
    // fond blanc
    {{{0, 501, 0}}, 500, {WHITE, BLACK, 0.0}},                
    // plafond blanc

    //SPHERES//
    {{{0, -0.5, -3}}, 0.5, {SKY, BLACK, 0.0}},        
    // boule bleue centrale (couleur ciel)

    {{{0, 1.4, -3}}, 0.5, {BLACK, WHITE, 30.0}},        
    // lumière

    // //LUMIERES//
    // {{{-1, 1, 1.1}}, 0.5, {BLACK, {{1.0, 0.6, 0.2}}, 5.0}},   
    // // LUMIERE (couleure noire, emission ORANGE)
    // {{{1, 1, 1.1}}, 0.5, {BLACK, {{0.7, 0.2, 1.0}}, 5.0}},   
    // // LUMIERE (couleure noire, emission VIOLETTE)
    // {{{-1, -1, 1.1}}, 0.5, {BLACK, {{0.55, 0.863, 1.0}}, 5.0}},   
    // // LUMIERE (couleure noire, emission CYAN)
    // {{{1, -1, 1.1}}, 0.5, {BLACK, {{0.431, 1.0, 0.596}}, 5.0}} 
    // // LUMIERE (couleure noire, emission VERT FLUO)
    };


    int nbSpheres = sizeof(sphere_list) / sizeof(sphere_list[0]);

    HitInfo closestHit;
    closestHit.didHit=false;
    closestHit.dst=INFINITY; // rien touché pour l'instant

    for(int i=0; i < nbSpheres ; i++){
        sphere s = sphere_list[i];
        HitInfo hitInfo =  hit_sphere(s.center, s.radius, r);

        if (hitInfo.didHit && hitInfo.dst < closestHit.dst){
            closestHit = hitInfo;
            closestHit.mat = s.mat;
        }
    }
    return closestHit;
}

__device__ color tracer(ray r, int nbRebondMax, curandState* globalState, int ind){

    color incomingLight = BLACK;
    color rayColor = WHITE;

    for (int i = 0; i<nbRebondMax; i++){

        HitInfo hitInfo = closest_hit(r);

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

__host__ __device__ void render_normal_albedo(vec3* normal, color* albedo, int hauteur_image, int largeur_image, camera cam){
    for (int j = hauteur_image - 1; j >= 0; j--){
        for (int i = 0; i < largeur_image; i++){
            int pixel_index = j*largeur_image+i;

            double u = (double)i/(largeur_image-1);
            double v = (double)j/(hauteur_image-1);

            ray r = get_ray(u, v, cam);
            HitInfo hitInfo = closest_hit(r);

            if (hitInfo.didHit){
                normal[pixel_index] = hitInfo.normal;

                if (hitInfo.mat.emissionStrength > 0.0){
                    albedo[pixel_index] = WHITE;
                }
                else{
                    albedo[pixel_index] = hitInfo.mat.diffuseColor;
                }
            }
            else{
                normal[pixel_index] = BLACK;
                albedo[pixel_index] = BLACK;
            }
        }
    }
}

__global__ void render_canva(color* canva, int largeur_image, int hauteur_image, int nbRayonParPixel, int nbRebondMax, camera cam, curandState* states) {
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
            totalLight = add(totalLight, tracer(r, nbRebondMax, states, ind));
        }

        canva[pixel_index] = write_color_canva(totalLight, nbRayonParPixel);
    }
}

int main(){

    // CONSTANTES (paramètres de rendu)
    ////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////

    double ratio = 4.0 / 3.0;
    int largeur_image = 1000;
    int hauteur_image = (int)(largeur_image / ratio);

    int nbRayonParPixel = 2000;
    int nbRebondMax = 6;
    
    int nbThreadsX = 8; // peut dépendre des GPU
    int nbThreadsY = 8; 

    ////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////
    
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

    sprintf(nomFichier, "CUDA_%dRAYS_%dRB_%02d-%02d_%02dh%02d.ppm", nbRayonParPixel, nbRebondMax-1, temps->tm_mday, temps->tm_mon + 1, temps->tm_hour, temps->tm_min);


    FILE *fichier = fopen(nomFichier, "w");

    // camera
    camera cam = init_camera(ratio);

    // tableau pour avoir chaque valeur de pixel au bon endroit (multithread et CUDA du coup)
    color* canva = (color*)malloc((largeur_image * hauteur_image)*sizeof(color));
    for (int i = 0; i < largeur_image*hauteur_image; i++) {
        canva[i] = BLACK;
    }

    // tableau pour avoir chaque valeur de pixel au bon endroit, avant rebond de lumière (necessaire au denoise)
    color* albedo = (color*)malloc((largeur_image * hauteur_image)*sizeof(color));
    for (int i = 0; i < largeur_image*hauteur_image; i++) {
        albedo[i] = BLACK;
    }

    // tableau pour avoir la normale de chaque objet (necessaire au denoise)
    color* normal = (color*)malloc((largeur_image * hauteur_image)*sizeof(color));
    for (int i = 0; i < largeur_image*hauteur_image; i++) {
        normal[i] = BLACK;
    }

    render_normal_albedo(normal, albedo, hauteur_image, largeur_image, cam);

    // défini la taille des blocks et threads
    dim3 blocks(largeur_image/nbThreadsX+1, hauteur_image/nbThreadsY+1);
    dim3 threads(nbThreadsX, nbThreadsY);

    // alloue la mémoire pour states sur le device (gpu) (necessaire à la fontion de random)
    curandState* states;
    cudaMalloc((void**) &states, (largeur_image * hauteur_image) * sizeof(curandState));

    // alloue la mémoire pour canva sur le device (gpu)
    color* canva_device;
    cudaMalloc((void**)&canva_device, (largeur_image * hauteur_image)*sizeof(color));

    // initialise les "states" pour la fonction de random
    init_curand_state<<<blocks, threads>>>(states, largeur_image, hauteur_image);

    // lance le rendu de canva
    render_canva<<<blocks, threads>>>(canva_device, largeur_image, hauteur_image, nbRayonParPixel, nbRebondMax, cam, states);

    // copie canva du device (gpu) vers l'host (cpu), puis free la mémoire de canva sur device
    cudaMemcpy(canva, canva_device, (largeur_image * hauteur_image)*sizeof(color), cudaMemcpyDeviceToHost);
    cudaFree(canva_device);
    
    // initialise le device OIDN
    
    OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT); // CPU or GPU if available
    oidnCommitDevice(device);

    OIDNBuffer colorBuf  = oidnNewBuffer(device, largeur_image * hauteur_image * 3 * sizeof(float));
    OIDNBuffer albedoBuf = oidnNewBuffer(device, largeur_image * hauteur_image * 3 * sizeof(float));
    OIDNBuffer normalBuf = oidnNewBuffer(device, largeur_image * hauteur_image * 3 * sizeof(float));

    float* floatImage = (float*)oidnGetBufferData(colorBuf);
    for(int i = 0; i < largeur_image * hauteur_image; i++) {
        floatImage[3*i] = (float)canva[i].e[0] / 255.0f;
        floatImage[3*i+1] = (float)canva[i].e[1] / 255.0f;
        floatImage[3*i+2] = (float)canva[i].e[2] / 255.0f;
    }

    float* floatAlbedo = (float*)oidnGetBufferData(albedoBuf);
    for(int i = 0; i < largeur_image * hauteur_image; i++) {
        floatAlbedo[3*i] = (float)albedo[i].e[0];
        floatAlbedo[3*i+1] = (float)albedo[i].e[1];
        floatAlbedo[3*i+2] = (float)albedo[i].e[2];
    }

    float* floatNormal = (float*)oidnGetBufferData(normalBuf);
    for(int i = 0; i < largeur_image * hauteur_image; i++) {
        floatNormal[3*i] = (float)normal[i].e[0];
        floatNormal[3*i+1] = (float)normal[i].e[1];
        floatNormal[3*i+2] = (float)normal[i].e[2];
    }

    // crée le filtre ("RT" est pour raytracing)
    OIDNFilter filter = oidnNewFilter(device, "RT");
    oidnSetFilterImage(filter, "color",  colorBuf, OIDN_FORMAT_FLOAT3, largeur_image, hauteur_image, 0, 0, 0); // beauty
    oidnSetFilterImage(filter, "albedo", albedoBuf, OIDN_FORMAT_FLOAT3, largeur_image, hauteur_image, 0, 0, 0); // auxiliary
    oidnSetFilterImage(filter, "normal", normalBuf, OIDN_FORMAT_FLOAT3, largeur_image, hauteur_image, 0, 0, 0); // auxiliary
    oidnSetFilterImage(filter, "output", colorBuf, OIDN_FORMAT_FLOAT3, largeur_image, hauteur_image, 0, 0, 0); // denoised beauty
    oidnCommitFilter(filter);

    // applique le filtre
    oidnExecuteFilter(filter);

    // check les erreurs du denoise
    const char* errorMessage;
    if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE){
        printf("Error: %s\n", errorMessage);
    }

    // retransfère les valeurs dans canva
    for (int i = 0; i < largeur_image * hauteur_image; i++) {
        canva[i].e[0] = (int)(floatImage[3*i] * 255.0f);
        canva[i].e[1] = (int)(floatImage[3*i+1] * 255.0f);
        canva[i].e[2] = (int)(floatImage[3*i+2] * 255.0f);
    }

    oidnReleaseBuffer(colorBuf);
    oidnReleaseBuffer(albedoBuf);
    oidnReleaseBuffer(normalBuf);
    oidnReleaseFilter(filter);
    oidnReleaseDevice(device);

    //base_ppm et canva_to_ppm réecrit ici pour contrer l'appel de fprintf impossible depuis une fonction __host__ __device__
    fprintf(fichier, "P3\n%d %d\n255\n", largeur_image, hauteur_image);

    for (int j = hauteur_image-1; j >= 0  ; j--){ 
        for (int i = 0; i < largeur_image; i++){
            fprintf(fichier, "%d %d %d\n", (int)canva[j*largeur_image+i].e[0], (int)canva[j*largeur_image+i].e[1], (int)canva[j*largeur_image+i].e[2]);
        }
    }
    
    fclose(fichier);

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

	return 0;
}
