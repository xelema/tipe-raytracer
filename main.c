#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#include <OpenImageDenoise/oidn.h>

#include "vec3.h"
#include "ray.h"
#include "hitinfo.h"
#include "sphere.h"
#include "rtutility.h"
#include "camera.h"
#include "denoiser.h"
#include "mesh.h"
#include "texture.h"

struct ThreadData {
    int start_row, end_row;
    color* canva;
    color* albedo_tab; 
    color* normal_tab;
    color* tex_list;
    material* mat_list;
    camera cam;
    
    int largeur_image, hauteur_image;
    int tex_width, tex_height;
    int nbRayonParPixel, nbRebondMax;
    int total_pixels;
    sphere* sphere_list;
    triangle* triangle_list;
    int nbSpheres, nbTriangles;

    int ouverture_x, ouverture_y, focus_distance;
    int AO_intensity;
    bool useAO;

};

int rendered_pixels = 0;

///////////////////////////////////////////////////////////////////////////////////////////////

HitInfo closest_hit(ray r, sphere* sphere_list, int nbSpheres, triangle* triangle_list, int nbTriangles, material* mat_list, int tex_width, int tex_height){

    HitInfo closestHit;
    closestHit.didHit=false;
    closestHit.dst=INFINITY; // rien touché pour l'instant

    // vérifie les intersections avec les sphères
    for(int i=0; i < nbSpheres ; i++){
        sphere s = sphere_list[i];
        HitInfo hitInfo = hit_sphere(s.center, s.radius, r);

        if (hitInfo.didHit && hitInfo.dst < closestHit.dst){
            closestHit = hitInfo;
            closestHit.mat = s.mat;
            closestHit.mat.alpha = 1.0;
        }
    }

    // vérifie les intersections avec les triangles
    for(int i=0; i < nbTriangles ; i++){
        triangle tri = triangle_list[i];
        HitInfo hitInfo = hit_triangle(tri, r);

        if (hitInfo.didHit && hitInfo.dst < closestHit.dst){
            material MaterialTex = get_material_from_matlist(tri, hitInfo, mat_list, tex_width, tex_height);
            closestHit = hitInfo;
            closestHit.mat = tri.mat;
            closestHit.mat.diffuseColor = MaterialTex.diffuseColor;
            closestHit.mat.alpha = MaterialTex.alpha;
        }
    }
    return closestHit;
}

color ambient_occlusion(vec3 point, vec3 normal, sphere* sphere_list, int nbSpheres, triangle* triangle_list, int nbTriangles, double AO_intensity, material* mat_list, int tex_width, int tex_height) {
    const int nbSamples = 1; // pour l'instant 1 suffit, à voir pour d'autres scènes

    color occlusion = BLACK;

    for (int i = 0; i < nbSamples; ++i) {
        vec3 randomDir = random_dir_no_norm();
        vec3 hemisphereDir = add(normal, randomDir);
        ray occlusionRay = {point, vec3_normalize(hemisphereDir)};

        HitInfo occlusionHit = closest_hit(occlusionRay, sphere_list, nbSpheres, triangle_list, nbTriangles, mat_list, tex_width, tex_height);

        if (occlusionHit.didHit) {
            double distance = vec3_length(sub(occlusionHit.hitPoint, point));
            double attenuation = distance / occlusionHit.dst;
            attenuation = pow(attenuation, AO_intensity);

            occlusion = add(occlusion, vec3_create(attenuation, attenuation, attenuation));
        }
    }

    return divide_scalar(divide_scalar(occlusion, nbSamples), AO_intensity);
}

col_alb_norm tracer(ray r, int nbRebondMax, sphere* sphere_list, int nbSpheres, triangle* triangle_list, int nbTriangles, double AO_intensity, bool useAO, material* mat_list, int tex_width, int tex_height){

    // cas des lumières
    HitInfo hitInfo = closest_hit(r, sphere_list, nbSpheres, triangle_list, nbTriangles, mat_list, tex_width, tex_height); 
    if (hitInfo.didHit){
        if (hitInfo.mat.emissionStrength > 0){
            color HSL = rgb_to_hsl(hitInfo.mat.emissionColor);
            HSL.e[2] *= 1.20; // luminosité (valeurs subjectives)
            HSL.e[1] *= 1.20; // saturation (valeurs subjectives)
            color newCol = hsl_to_rgb(HSL);
            return can_create(newCol, hitInfo.mat.emissionColor, hitInfo.normal);
        }
    }  

    // pas une lumière
    color incomingLight = BLACK;
    color rayColor = WHITE;
    color albedo_color = BLACK; // denoiser
    color normal_color = BLACK; // denoiser
    bool is_alpha = false; // denoiser

    for (int i = 0; i<nbRebondMax; i++){

        HitInfo hitInfo = closest_hit(r, sphere_list, nbSpheres, triangle_list, nbTriangles, mat_list, tex_width, tex_height);
        
        if (i==0){ // denoiser cas de base
            albedo_color = hitInfo.mat.diffuseColor;
            normal_color = hitInfo.normal;
        }
        if (i==1 && is_alpha){ // denoiser cas alpha (trou dans la texture)
            albedo_color = hitInfo.mat.diffuseColor;
            normal_color = hitInfo.normal;
            is_alpha = false;
        }
        if (hitInfo.didHit){   
            if(hitInfo.mat.alpha > 0.9){
                material mat = hitInfo.mat;

                r.origin = hitInfo.hitPoint;
                vec3 diffuse_dir = vec3_normalize(add(hitInfo.normal,random_dir_no_norm())); // sebastian lague
                vec3 reflected_dir = reflected_vec(r.dir, hitInfo.normal);
                r.dir = vec3_lerp(diffuse_dir, reflected_dir, mat.reflectionStrength);

                if (useAO){ // calcul avec occlusion ambiante (notamment augmentation des lumières)

                    color emittedLight = multiply_scalar(mat.emissionColor, mat.emissionStrength * 1.5 * AO_intensity);

                    incomingLight = add(incomingLight,multiply(emittedLight, rayColor));
                    rayColor = multiply(mat.diffuseColor, rayColor);

                    color occlusion = ambient_occlusion(hitInfo.hitPoint, hitInfo.normal, sphere_list, nbSpheres, triangle_list, nbTriangles, AO_intensity, mat_list, tex_width, tex_height);
                    rayColor = multiply(rayColor, occlusion); // applique l'occlusion ambiante à la couleur du rayon
                }

                else{  // calcul sans occlusion ambiante

                    color emittedLight = multiply_scalar(mat.emissionColor, mat.emissionStrength);

                    incomingLight = add(incomingLight,multiply(emittedLight, rayColor));
                    rayColor = multiply(mat.diffuseColor, rayColor);
                }
            }
            else{
                r.origin = hitInfo.hitPoint;
                is_alpha = true;
                continue;
            }
        }
        else{
            break;
        }
    }
    return can_create(incomingLight, albedo_color, normal_color);
}



void* fill_canva(void *arg) {
    struct ThreadData* data = (struct ThreadData*)arg;

    for (int j = data->start_row; j >= data->end_row; --j) {

        // debug (affichage en %)
        rendered_pixels += data->largeur_image;

        int update_frequency = data->total_pixels / 40;
        if (rendered_pixels % update_frequency == 0) {
            int percentage = (rendered_pixels * 100) / data->total_pixels;
            fprintf(stderr, "Progression : %d%%\n", percentage);
            fflush(stderr); 
        }

        for (int i = 0; i < data->largeur_image; i++) {
            int pixel_index = j*data->largeur_image+i;
            col_alb_norm totalLight = {{BLACK, BLACK, BLACK}};
            
            for (int x = 0; x < data->nbRayonParPixel; ++x) {
                double u = ((double)i + randomDouble(-0.5, 0.5))/(data->largeur_image-1);
                double v = ((double)j + randomDouble(-0.5, 0.5))/(data->hauteur_image-1);

                double dx_ouverture = randomDouble(-0.5, 0.5) * data->ouverture_x;
                double dy_ouverture = randomDouble(-0.5, 0.5) * data->ouverture_y;

                ray r = get_ray(u, v, data->cam, data->focus_distance, dx_ouverture, dy_ouverture);
                totalLight = add_col_alb_norm(totalLight, tracer(r, data->nbRebondMax, data->sphere_list, data->nbSpheres, data->triangle_list, data->nbTriangles, data->AO_intensity, data->useAO, data->mat_list, data->tex_width, data->tex_height)); 
            }

            data->canva[pixel_index] = write_color_canva(totalLight.e[0], data->nbRayonParPixel);

            // for denoiser
            data->albedo_tab[pixel_index] = divide_scalar(totalLight.e[1], data->nbRayonParPixel);
            data->normal_tab[pixel_index] = divide_scalar(totalLight.e[2], data->nbRayonParPixel);
        }
    }

    pthread_exit(NULL);
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
    
    point3 origin = {{0.34, 0.3, 0.5}}; // position de la camera
    point3 target = {{0.0, -0.5, -3}}; // cible de la camera
    vec3 up = {{0, 1, 0}}; // permet de modifier la rotation selon l'axe z ({{0, 1, 0}} pour horizontal)

    double focus_distance = 3; // distance de mise au point (depth of field)
    double ouverture_x = 0.0; // quantité de dof horizontal
    double ouverture_y = 0.0; // quantité de dof vertical

    //qualité et performance
    int nbRayonParPixel = 200;
    int nbRebondMax = 5;
    
    #define NUM_THREADS 16

    bool useDenoiser = true;

    bool useAO = true; // occlusion ambiante, rendu environ 2x plus lent
    double AO_intensity = 2.5; // supérieur à 1 pour augmenter l'intensité

    // chemin des fichiers de mesh
    char* obj_file = "model3D/1st_mc_world/tree/tree_tri_less.obj"; // chemin du fichier obj
    char* tex_file = "model3D/1st_mc_world/tree/tex.ppm"; // chemin du fichier de texture (converti en PPM P3)
    char* alpha_file = "model3D/1st_mc_world/tree/tex_alpha.ppm"; // chemin du fichier d'alpha (converti en PPM P3)

    // nom du fichier de sortie
    char nomFichier[100];
    time_t maintenant = time(NULL); // heure actuelle pour le nom du fichier
    struct tm *temps = localtime(&maintenant);
    sprintf(nomFichier, "world_alpha_%dRAYS_%dRB_%02d-%02d_%02dh%02d.ppm", nbRayonParPixel, nbRebondMax-1, temps->tm_mday, temps->tm_mon + 1, temps->tm_hour, temps->tm_min);

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
        {{{0.4, -0.5, -3.3}}, 0.5, {SKY, BLACK, 0.0, 0.999}},
        // {{{0.2, -0.7, -2}}, 0.3, {SKY, BLACK, 0.0, 0.5}},
    };

    ////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////

    // temps d'execution
    struct timeval start_time, end_time;
    gettimeofday (&start_time, NULL);

    // // liste de material pour la texture du mesh
    int tex_width, tex_height;
    material* mat_list = create_mat_list(tex_file, alpha_file, &tex_width, &tex_height);

    // liste de triangle qui forment un mesh (a partir d'un fichier obj)
    int nbTriangles;
    triangle* mesh_list = list_of_mesh(obj_file, &nbTriangles);
    move_mesh(-0.4, -1.0, -2, &mesh_list, nbTriangles); // translation (x, y, z) du mesh

    int total_pixels = largeur_image * hauteur_image;
    int nbSpheres = sizeof(sphere_list) / sizeof(sphere_list[0]);

    FILE *fichier = fopen(nomFichier, "w");

    // camera
    camera cam = init_camera(origin, target, up, vfov, ratio);

    // tableau pour avoir chaque valeur de pixel au bon endroit (multithread)
    color* canva = (color*)malloc((largeur_image*hauteur_image)*sizeof(struct Vec3));
    for (int i = 0; i < largeur_image*hauteur_image; i++) {
        canva[i] = (color)BLACK;
    }

    // tableau pour avoir chaque valeur de pixel au bon endroit, avant rebond de lumière (necessaire au denoise)
    color* albedo_tab = (color*)malloc((largeur_image * hauteur_image)*sizeof(color));
    for (int i = 0; i < largeur_image*hauteur_image; i++) {
        albedo_tab[i] = (color)BLACK;
    }

    // tableau pour avoir la normale de chaque objet (necessaire au denoise)
    color* normal_tab = (color*)malloc((largeur_image * hauteur_image)*sizeof(color));
    for (int i = 0; i < largeur_image*hauteur_image; i++) {
        normal_tab[i] = (color)BLACK;
    }
    
    // création des threads

    pthread_t threads[NUM_THREADS];
    struct ThreadData thread_data[NUM_THREADS];

    int rows_per_thread = hauteur_image / NUM_THREADS;
    int remaining_rows = hauteur_image % NUM_THREADS;
    int start_row = hauteur_image - 1;

    for (int i = 0; i < NUM_THREADS; i++) {
        int end_row = start_row - rows_per_thread + 1;

        if (i == NUM_THREADS - 1) {
            end_row -= remaining_rows;
        }

        thread_data[i].start_row = start_row;
        thread_data[i].end_row = end_row;
        thread_data[i].canva = canva;
        thread_data[i].cam = cam;
        thread_data[i].largeur_image = largeur_image;
        thread_data[i].hauteur_image = hauteur_image;
        thread_data[i].nbRayonParPixel = nbRayonParPixel;
        thread_data[i].nbRebondMax = nbRebondMax;
        thread_data[i].sphere_list = sphere_list;
        thread_data[i].nbSpheres = nbSpheres;
        thread_data[i].total_pixels = total_pixels;
        thread_data[i].AO_intensity = AO_intensity;
        thread_data[i].useAO = useAO;
        thread_data[i].ouverture_x = ouverture_x;
        thread_data[i].ouverture_y = ouverture_y;
        thread_data[i].focus_distance = focus_distance;
        thread_data[i].nbTriangles = nbTriangles;
        thread_data[i].triangle_list = mesh_list;
        thread_data[i].albedo_tab = albedo_tab;
        thread_data[i].normal_tab = normal_tab;
        thread_data[i].mat_list = mat_list;
        thread_data[i].tex_height = tex_height;
        thread_data[i].tex_width = tex_width;

        pthread_create(&threads[i], NULL, fill_canva, (void*)&thread_data[i]);

        start_row = end_row - 1;
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    if (useDenoiser) denoiser(largeur_image, hauteur_image, canva, cam, albedo_tab, normal_tab);

    fprintf(fichier, "P3\n%d %d\n255\n", largeur_image, hauteur_image);

    // rendu normal
    for (int j = hauteur_image-1; j >= 0  ; j--){ 
        for (int i = 0; i < largeur_image; i++){
            int pixel_index = j*largeur_image+i;
            fprintf(fichier, "%d %d %d\n", (int)(canva[pixel_index].e[0]), (int)(canva[pixel_index].e[1]), (int)(canva[pixel_index].e[2]));
        }
    }

    // rendu albedo
    // for (int j = hauteur_image-1; j >= 0  ; j--){ 
    //     for (int i = 0; i < largeur_image; i++){
    //         int pixel_index = j*largeur_image+i;
    //         fprintf(fichier, "%d %d %d\n", (int)(albedo_tab[pixel_index].e[0]*255), (int)(albedo_tab[pixel_index].e[1]*255), (int)(albedo_tab[pixel_index].e[2]*255));
    //     }
    // }

    // rendu normales
    // for (int j = hauteur_image-1; j >= 0  ; j--){ 
    //     for (int i = 0; i < largeur_image; i++){
    //         int pixel_index = j*largeur_image+i;
    //         fprintf(fichier, "%d %d %d\n", (int)((normal_tab[pixel_index].e[0]*0.5+0.5)*255), (int)((normal_tab[pixel_index].e[1]*0.5+0.5)*255), (int)((normal_tab[pixel_index].e[2]*0.5+0.5)*255));
    //     }
    // }

    fclose(fichier);    
    free(canva);
    free(albedo_tab);
    free(normal_tab);
    free(mesh_list);
    // free(tex_list);
    free(mat_list);

    gettimeofday(&end_time, NULL);

    long seconds = end_time.tv_sec - start_time.tv_sec;

    fprintf(stderr, "Fini.\n");
    fprintf(stderr, "\nTemps d'exécution : %ld min %ld sec\n", seconds / 60, seconds % 60);

    return 0;
}