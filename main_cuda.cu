#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#include <OpenImageDenoise/oidn.h>
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/color4.h>

#include "vec3.hu"
#include "ray.hu"
#include "hitinfo.hu"
#include "sphere.hu"
#include "rtutility.hu"
#include "camera.hu"
#include "denoiser.hu"
#include "triangle.hu"

__host__ __device__ HitInfo closest_hit(ray r, sphere* spheres, int nbSpheres, triangle* triangles, int nbTriangles){

    HitInfo closestHit;
    closestHit.didHit=false;
    closestHit.dst=INFINITY; // rien touché pour l'instant

    // vérifie les intersections avec les sphères
    for(int i=0; i < nbSpheres ; i++){
        sphere s = spheres[i];
        HitInfo hitInfo = hit_sphere(s.center, s.radius, r);

        if (hitInfo.didHit && hitInfo.dst < closestHit.dst){
            closestHit = hitInfo;
            closestHit.mat = s.mat;
        }
    }

    // vérifie les intersections avec les triangles
    for(int i=0; i < nbTriangles ; i++){
        triangle tri = triangles[i];
        HitInfo hitInfo = hit_triangle(tri, r);

        if (hitInfo.didHit && hitInfo.dst < closestHit.dst){
            closestHit = hitInfo;
            closestHit.mat = tri.mat;
        }
    }
    return closestHit;
}

__device__ color ambient_occlusion(vec3 point, vec3 normal, sphere* spheres, int nbSpheres, triangle* triangles, int nbTriangles, curandState* globalState, int ind, double AO_intensity) {
    const int nbSamples = 1; // pour l'instant 1 suffit, à voir pour d'autres scènes

    color occlusion = BLACK;

    for (int i = 0; i < nbSamples; ++i) {
        vec3 randomDir = random_dir_no_norm(globalState, ind);
        vec3 hemisphereDir = add(normal, randomDir);
        ray occlusionRay = {point, vec3_normalize(hemisphereDir)};

        HitInfo occlusionHit = closest_hit(occlusionRay, spheres, nbSpheres, triangles, nbTriangles);

        if (occlusionHit.didHit) {
            double distance = vec3_length(sub(occlusionHit.hitPoint, point));
            double attenuation = distance / occlusionHit.dst;
            attenuation = pow(attenuation, AO_intensity);

            occlusion = add(occlusion, {{attenuation, attenuation, attenuation}});
        }
    }

    return divide_scalar(divide_scalar(occlusion, nbSamples), AO_intensity);
}


__device__ col_alb_norm tracer(ray r, int nbRebondMax, curandState* globalState, int ind, sphere* spheres, int nbSpheres, triangle* triangles, int nbTriangles, double AO_intensity, bool useAO){

    // cas des lumières
    HitInfo hitInfo = closest_hit(r, spheres, nbSpheres, triangles, nbTriangles); 
    if (hitInfo.didHit){
        if (hitInfo.mat.emissionStrength > 0){
            color HSL = rgb_to_hsl(hitInfo.mat.emissionColor);
            HSL.e[2] *= 1.20; // luminosité
            HSL.e[1] *= 1.20; // saturation (valeurs subjectives)
            color newCol = hsl_to_rgb(HSL);
            return {newCol, newCol, hitInfo.normal};
        }
    }
    else return BLACK;
    
    // pas une lumière
    color incomingLight = BLACK;
    color rayColor = WHITE;

    for (int i = 0; i<nbRebondMax; i++){

        HitInfo hitInfo = closest_hit(r, spheres, nbSpheres, triangles, nbTriangles);

        if (hitInfo.didHit){
            material mat = hitInfo.mat;

            r.origin = hitInfo.hitPoint;
            vec3 diffuse_dir = vec3_normalize(add(hitInfo.normal,random_dir_no_norm(globalState, ind))); // sebastian lague
            vec3 reflected_dir = sub(r.dir, multiply_scalar(hitInfo.normal, 2*vec3_dot(r.dir, hitInfo.normal)));
            r.dir = vec3_lerp(diffuse_dir, reflected_dir, mat.reflectionStrength);

            if (useAO){ // calcul avec occlusion ambiante (notamment augmentation des lumières)

                color emittedLight = multiply_scalar(mat.emissionColor, mat.emissionStrength * AO_intensity);

                incomingLight = add(incomingLight,multiply(emittedLight, rayColor));
                rayColor = multiply(mat.diffuseColor, rayColor);

                color occlusion = ambient_occlusion(hitInfo.hitPoint, hitInfo.normal, spheres, nbSpheres, triangles, nbTriangles, globalState, ind, AO_intensity);
                rayColor = multiply(rayColor, occlusion); // applique l'occlusion ambiante à la couleur du rayon
            }

            else{  // calcul sans occlusion ambiante

                color emittedLight = multiply_scalar(mat.emissionColor, mat.emissionStrength);

                incomingLight = add(incomingLight,multiply(emittedLight, rayColor));
                rayColor = multiply(mat.diffuseColor, rayColor);
            }
        }
        else{
            break;
        }
    }
    return {incomingLight, hitInfo.mat.diffuseColor, hitInfo.normal};
}


__global__ void render_canva(color* canva, int largeur_image, int hauteur_image, int nbRayonParPixel, int nbRebondMax, camera cam, curandState* states, sphere* spheres, int nbSpheres, triangle* triangles, int nbTriangles, double AO_intensity, bool useAO, double focus_distance, double ouverture_x, double ouverture_y, color* normal_tab, color* albedo_tab) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = j * gridDim.x * blockDim.x + i;

    if (i < largeur_image && j < hauteur_image){
        int pixel_index = j*largeur_image+i;
        col_alb_norm totalLight = {{BLACK, BLACK, BLACK}};

        for (int k=0; k<nbRayonParPixel; ++k){
 
            double u = ((double)i + 0.5 + randomDouble(states, ind, -0.5, 0.5))/(largeur_image-1);
            double v = ((double)j + 0.5 + randomDouble(states, ind, -0.5, 0.5))/(hauteur_image-1);
            
            double dx_ouverture = randomDouble(states, ind, -0.5, 0.5) * ouverture_x;
            double dy_ouverture = randomDouble(states, ind, -0.5, 0.5) * ouverture_y;

            ray r = get_ray(u, v, cam, focus_distance, dx_ouverture, dy_ouverture);
            totalLight = add_col_alb_norm(totalLight, tracer(r, nbRebondMax, states, ind, spheres, nbSpheres, triangles, nbTriangles, AO_intensity, useAO)); 
        }

        // for denoiser
        canva[pixel_index] = write_color_canva(totalLight.e[0], nbRayonParPixel);
        albedo_tab[pixel_index] = divide_scalar(totalLight.e[1], nbRayonParPixel);
        normal_tab[pixel_index] = divide_scalar(totalLight.e[2], nbRayonParPixel);

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
    double vfov = 70; // fov vertical en degrée
    
    point3 origin = {{-0.7, 0, 0}}; // position de la camera
    point3 target = {{0.3, -0.5, -3}}; // cible de la camera
    vec3 up = {{0, 1, 0}}; // permet de modifier la rotation selon l'axe z ({{0, 1, 0}} pour horizontal)

    double focus_distance = 3; // distance de mise au point (depth of field)
    double ouverture_x = 0.;
    double ouverture_y = 0.;

    //qualité et performance
    int nbRayonParPixel = 10;
    int nbRebondMax = 5;
    
    int nbThreadsX = 8; // peut dépendre des GPU
    int nbThreadsY = 8; 

    bool useDenoiser = false;

    bool useAO = false; // occlusion ambiante, rendu environ 2x plus lent
    double AO_intensity = 3; // supérieur à 1 pour augmenter l'intensité

    //position des sphères dans la scène
    sphere h_sphere_list[10] = {
        //{position du centre x, y, z}, rayon, {couleur de l'objet, couleur d'emission, intensité d'emission (> 1), intensité de reflection (entre 0 et 1)}
        {{{-501,0,0}}, 500, {GREEN, BLACK, 0.0, 0.96}},
        {{{0,-501,0}}, 500, {WHITE, BLACK, 0.0, 0.4}},
        {{{501, 0, 0}}, 500, {RED, BLACK, 0.0, 0.96}},
        {{{-0.5, 1.4, -3.}}, 0.5, {BLACK, {{1.0, 0.6, 0.2}}, 8.0, 0.0}},
        {{{0.5, 1.4, -2.}}, 0.5, {BLACK, {{0.7, 0.2, 1.0}}, 8.0, 0.0}},
        {{{-0.5, -1.4, -1.5}}, 0.5, {BLACK, {{0.55, 0.863, 1.0}}, 4.5, 0.0}},
        {{{0.5, -1.4, -3.1}}, 0.5, {BLACK, {{0.431, 1.0, 0.596}}, 4.5, 0.0}},
        {{{0, 0, -504}}, 500, {WHITE, BLACK, 0.0, 0.0}},
        {{{0, 501, 0}}, 500, {WHITE, BLACK, 0.0, 0.0}},
        {{{-0.4, -0.5, -3.3}}, 0.5, {SKY, BLACK, 0.0, 1.0}},
        // {{{0.2, -0.7, -2}}, 0.3, {SKY, BLACK, 0.0, 0.5}},
    };

    // triangle h_triangle_list[2] = {
    //     // {point A, point B, point C, {couleur de l'objet, couleur d'émission, intensité d'émission (> 1), intensité de réflexion (entre 0 et 1)}}
    //     {{-0.8, -0.8, -3.2}, {0.8, -0.8, -3.2}, {0, 0.8, -3.2}, {BLUE, BLACK, 0.0, 0.5}},
    //     {{-0.5, -0.2, -2}, {0.4, -0.7, -2}, {0, 0.8, -2}, {{1.0, 0., 1.0}, BLACK, 0.0, 0.5}},
    // };

    // nom du fichier
    char nomFichier[100];
    time_t maintenant = time(NULL); // Obtenir l'heure actuelle
    struct tm *temps = localtime(&maintenant); // Convertir en structure tm

    sprintf(nomFichier, "MESH_%dRAYS_%dRB_%02d-%02d_%02dh%02d.ppm", nbRayonParPixel, nbRebondMax-1, temps->tm_mday, temps->tm_mon + 1, temps->tm_hour, temps->tm_min);

    ////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////

    // assimp part
    ////////////////////////////////////////////////////////

    const struct aiScene* scene = aiImportFile("1tree_tri.obj", aiProcess_Triangulate);

    if (scene == NULL) {
        printf("Error parsing: %s\n", aiGetErrorString());
        return false;
    }

    int total_triangles = 0;
    for (int i = 0; i < scene->mNumMeshes; i++) {
        const struct aiMesh* mesh = scene->mMeshes[i];
        total_triangles += mesh->mNumFaces;
    }
    printf("num triangles dans mesh : %d\n", total_triangles);

    triangle* h_triangle_list = (triangle*)malloc(total_triangles * sizeof(triangle));

    int triangle_index = 0;

    vec3 displacement = {{0.3, -1.1, -2.1}};

    for (int i = 0; i < scene->mNumMeshes; i++) {
        const struct aiMesh* mesh = scene->mMeshes[i];
        const struct aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

        // Get the diffuse color of the material
        aiColor4D aiColor;
        if (AI_SUCCESS == aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &aiColor)) {
            printf("Diffuse color: %f %f %f\n", aiColor.r, aiColor.g, aiColor.b);
        }

        color my_col = {{aiColor.r, aiColor.g, aiColor.b}};


        // Get the shininess of the material
        float shininess;
        if (AI_SUCCESS == aiGetMaterialFloat(material, AI_MATKEY_SHININESS, &shininess)) {
            printf("Shininess: %f\n", shininess);
        }

        for (int j = 0; j < mesh->mNumFaces; j++) {
            const struct aiFace face = mesh->mFaces[j];

            triangle* tri = &h_triangle_list[triangle_index];

            tri->A = add(aiVector3D_to_vec3(mesh->mVertices[face.mIndices[0]]), displacement);
            printf("%f %f %f   ", tri->A.e[0], tri->A.e[1], tri->A.e[2]);
            tri->B = add(aiVector3D_to_vec3(mesh->mVertices[face.mIndices[1]]), displacement);
            printf("%f %f %f   ", tri->B.e[0], tri->B.e[1], tri->B.e[2]);
            tri->C = add(aiVector3D_to_vec3(mesh->mVertices[face.mIndices[2]]), displacement);
            printf("%f %f %f\n", tri->C.e[0], tri->C.e[1], tri->C.e[2]);
            tri->mat = {my_col, BLACK, 0.0, (double)shininess};

            printf("index : %d\n", triangle_index);

            triangle_index++;
        }
    }

    printf("triangle 1, A : %f %f %f\n", h_triangle_list[1].A.e[0], h_triangle_list[1].A.e[1], h_triangle_list[1].A.e[2]);

    ////////////////////////////////////////////////////////

    int nbSpheres = sizeof(h_sphere_list) / sizeof(h_sphere_list[0]);
    int nbTriangles = total_triangles;

    printf("num triangles dans liste : %d\n", nbTriangles);
    FILE *fichier = fopen(nomFichier, "w");

    // temps d'execution
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // camera
    camera cam = init_camera(origin, target, up, vfov, ratio);

    // tableau pour avoir chaque valeur de pixel au bon endroit (multithread et CUDA du coup)
    color* canva = (color*)malloc((largeur_image * hauteur_image)*sizeof(color));
    for (int i = 0; i < largeur_image*hauteur_image; i++) {
        canva[i] = BLACK;
    }

    // tableau pour avoir chaque valeur de pixel au bon endroit, avant rebond de lumière (necessaire au denoise)
    color* albedo_tab = (color*)malloc((largeur_image * hauteur_image)*sizeof(color));
    for (int i = 0; i < largeur_image*hauteur_image; i++) {
        albedo_tab[i] = BLACK;
    }

    // tableau pour avoir la normale de chaque objet (necessaire au denoise)
    color* normal_tab = (color*)malloc((largeur_image * hauteur_image)*sizeof(color));
    for (int i = 0; i < largeur_image*hauteur_image; i++) {
        normal_tab[i] = BLACK;
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

    color* albedo_tab_device;
    cudaMalloc((void**)&albedo_tab_device, (largeur_image * hauteur_image)*sizeof(color));

    color* normal_tab_device;
    cudaMalloc((void**)&normal_tab_device, (largeur_image * hauteur_image)*sizeof(color));

    // alloue la mémoire pour d_sphere_list sur le device puis copie h_sphere_list (host) vers le device, optimisation
    sphere* d_sphere_list;
    cudaMalloc((void**)&d_sphere_list, nbSpheres*sizeof(sphere));
    cudaMemcpy(d_sphere_list, h_sphere_list, nbSpheres*sizeof(sphere), cudaMemcpyHostToDevice);
    
    // alloue la mémoire pour d_triangle_list sur le device puis copie h_triangle_list (host) vers le device, optimisation
    triangle* d_triangle_list;
    cudaMalloc((void**)&d_triangle_list, nbTriangles*sizeof(triangle));
    cudaMemcpy(d_triangle_list, h_triangle_list, nbTriangles*sizeof(triangle), cudaMemcpyHostToDevice);

    // initialise les "states" pour la fonction de random
    init_curand_state<<<blocks, threads>>>(states, largeur_image, hauteur_image);

    // lance le rendu de canva
    render_canva<<<blocks, threads>>>(canva_device, largeur_image, hauteur_image, nbRayonParPixel, nbRebondMax, cam, states, d_sphere_list, nbSpheres, d_triangle_list, nbTriangles, AO_intensity, useAO, focus_distance, ouverture_x, ouverture_y, normal_tab_device, albedo_tab_device);

    // copie canva du device (gpu) vers l'host (cpu), puis free la mémoire de canva sur device
    cudaMemcpy(canva, canva_device, (largeur_image * hauteur_image)*sizeof(color), cudaMemcpyDeviceToHost);
    cudaFree(canva_device);

    cudaMemcpy(albedo_tab, albedo_tab_device, (largeur_image * hauteur_image)*sizeof(color), cudaMemcpyDeviceToHost);
    cudaFree(albedo_tab_device);

    cudaMemcpy(normal_tab, normal_tab_device, (largeur_image * hauteur_image)*sizeof(color), cudaMemcpyDeviceToHost);
    cudaFree(normal_tab_device);

    // utilise le denoiser si l'option est activée
    if (useDenoiser) denoiser(largeur_image, hauteur_image, canva, cam, albedo_tab, normal_tab);
    
    //base_ppm et canva_to_ppm réecrit ici pour contrer l'appel de fprintf impossible depuis une fonction __host__ __device__
    fprintf(fichier, "P3\n%d %d\n255\n", largeur_image, hauteur_image);

    for (int j = hauteur_image-1; j >= 0  ; j--){ 
        for (int i = 0; i < largeur_image; i++){
            fprintf(fichier, "%d %d %d\n", (int)(canva[j*largeur_image+i].e[0]), (int)(canva[j*largeur_image+i].e[1]), (int)(canva[j*largeur_image+i].e[2]));
        }
    }
    
    fclose(fichier);
    cudaFree(d_sphere_list);
    cudaFree(d_triangle_list);
    cudaFree(states);

    aiReleaseImport(scene);
    free(h_triangle_list);

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