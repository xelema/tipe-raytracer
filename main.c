#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#include "vec3.h"
#include "ray.h"
#include "hitinfo.h"
#include "sphere.h"
#include "rtutility.h"
#include "camera.h"

const sphere sphere_list[10] = {
    {{{-501,0,0}}, 500, {GREEN, BLACK, 0.0}},                 
    // mur gauche vert
    {{{0,-501,0}}, 500, {WHITE, BLACK, 0.0}},                 
    // sol blanc
    {{{501, 0, 0}}, 500, {RED, BLACK, 0.0}},                  
    // mur droite rouge
    {{{-0.5, 1.4, -3}}, 0.5, {BLACK, {{1.0, 0.6, 0.2}}, 8.0}},   
    // LUMIERE (couleure noire, emission ORANGE)
    {{{0.5, 1.4, -3}}, 0.5, {BLACK, {{0.7, 0.2, 1.0}}, 8.0}},   
    // LUMIERE (couleure noire, emission VIOLETTE)
    {{{-0.5, -1.4, -3}}, 0.5, {BLACK, {{0.55, 0.863, 1.0}}, 5.0}},   
    // LUMIERE (couleure noire, emission CYAN)
    {{{0.5, -1.4, -3}}, 0.5, {BLACK, {{0.431, 1.0, 0.596}}, 5.0}},   
    // LUMIERE (couleure noire, emission VERT FLUO)
    {{{0, 0, -504}}, 500, {WHITE, BLACK, 0.0}},               
    // fond blanc
    {{{0, 501, 0}}, 500, {WHITE, BLACK, 0.0}},                
    // plafond blanc
    {{{0, 0, -3}}, 0.5, {SKY, BLACK, 0.0}}                    
    // boule bleue centrale (couleur ciel)
    };

struct ThreadData {
    int start_row;
    int end_row;
    color* canva;
    camera cam;
};

int total_pixels = largeur_image * hauteur_image;
int rendered_pixels = 0;

///////////////////////////////////////////////////////////////////////////////////////////////

HitInfo hit_sphere(point3 center, double radius, ray r){

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

HitInfo closest_hit(ray r){
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


point3 tracer(ray r){

    color incomingLight = BLACK;
    color rayColor = WHITE;

    for (int i = 0; i<nbRebondMax; i++){

        HitInfo hitInfo = closest_hit(r);

        if (hitInfo.didHit){
            r.origin = hitInfo.hitPoint;
            r.dir = random_dir(hitInfo.normal);

            material mat = hitInfo.mat;

            color emittedLight = multiply_scalar(mat.emissionColor, mat.emissionStrength);

            double lightStrength = vec3_dot(hitInfo.normal, r.dir); // Loi de Lambert

            incomingLight = add(incomingLight,multiply(emittedLight, rayColor));
            rayColor = multiply(multiply_scalar(mat.diffuseColor, lightStrength*2 /*trop sombre sinon*/ ), rayColor); 
        }
        else{
            break;
        }
    }
    return incomingLight;
}

color ray_color(ray r) {
    color t = tracer(r);
    return t;
}

void* fill_canva(void *arg) {
    struct ThreadData* data = (struct ThreadData*)arg;

    for (int j = data->start_row; j >= data->end_row; --j) {

        // debug (affichage en %)
        rendered_pixels += largeur_image;

        int update_frequency = total_pixels / 100;
        if (rendered_pixels % update_frequency == 0) {
            int percentage = (rendered_pixels * 100) / total_pixels;
            fprintf(stderr, "Progression : %d%%\n", percentage);
            fflush(stderr); 
        }

        for (int i = 0; i < largeur_image; i++) {
            color totalLight = BLACK;
            
            for (int x = 0; x < nbRayonParPixel; ++x) {
                double u = ((double)i + randomDouble(-0.5, 0.5))/(largeur_image-1);
                double v = ((double)j + randomDouble(-0.5, 0.5))/(hauteur_image-1);

                ray r = get_ray(u, v, data->cam);
                totalLight = add(totalLight, ray_color(r));
            }

            data->canva[j*largeur_image+i] = write_color_canva(totalLight);
        }
    }

    pthread_exit(NULL);
}

int main(){
    
    // temps d'execution
    struct timeval start_time, end_time;
    gettimeofday (&start_time, NULL);

    // nom du fichier
    char nomFichier[100];
    time_t maintenant = time(NULL); // Obtenir l'heure actuelle
    struct tm *temps = localtime(&maintenant); // Convertir en structure tm

    sprintf(nomFichier, "multithreading_%dRAYS_%dRB_%02d-%02d_%02dh%02d.ppm", nbRayonParPixel, nbRebondMax-1, temps->tm_mday, temps->tm_mon + 1, temps->tm_hour, temps->tm_min);


    FILE *fichier = fopen(nomFichier, "w");

    // camera
    camera cam = init_camera();

    // tableau pour avoir chaque valeur de pixel au bon endroit (multithread)
    color* canva = (color*)malloc((largeur_image*hauteur_image)*sizeof(struct Vec3));
    for (int i = 0; i < largeur_image*hauteur_image; i++) {
        canva[i] = (color)BLACK;
    }

    base_ppm(fichier);
    
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

        pthread_create(&threads[i], NULL, fill_canva, (void*)&thread_data[i]);

        start_row = end_row - 1;
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    canva_to_ppm(fichier, canva);
    fclose(fichier);

    gettimeofday(&end_time, NULL);

    long seconds = end_time.tv_sec - start_time.tv_sec;

    fprintf(stderr, "Fini.\n");
    fprintf(stderr, "\nTemps d'exécution : %ld min %ld sec\n", seconds / 60, seconds % 60);

	return 0;
}
