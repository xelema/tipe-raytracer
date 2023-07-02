#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include "vec3.h"
#include "ray.h"
#include "hitinfo.h"
#include "sphere.h"
#include "rtutility.h"

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
            r.dir = bon_sens(hitInfo.normal);

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

int main(){
    
    // temps d'execution
    struct timeval start_time, end_time;
    gettimeofday (&start_time, NULL);

    // nom du fichier
    char nomFichier[100];
    time_t maintenant = time(NULL); // Obtenir l'heure actuelle
    struct tm *temps = localtime(&maintenant); // Convertir en structure tm

    sprintf(nomFichier, "newscene_lambertslaw_%dRAYS_%dRB_%02d-%02d_%02dh%02d.ppm", nbRayonParPixel, nbRebondMax-1, temps->tm_mday, temps->tm_mon + 1, temps->tm_hour, temps->tm_min);


    FILE *fichier = fopen(nomFichier, "w");

    // camera
    double hauteur_viewport = 1.0;
    double largeur_viewport = ratio*hauteur_viewport;
    double focal_length = 1.0;

    point3 origin = {{0, 0, 0}};
    vec3 horizontal = {{largeur_viewport, 0, 0}};
    vec3 vertical = {{0, hauteur_viewport, 0}};
    vec3 focal_length_vec = {{0, 0, focal_length}};
    vec3 coin_bas_gauche = sub(origin, add(divide(horizontal,2), add(divide(vertical, 2), focal_length_vec))); 
    //coin_bas_gauche = origin - horizontal/2 - vertical/2 - profondeur

    
    // render
    base_ppm(fichier);
    
    for (int j = hauteur_image - 1; j >= 0  ; --j) {
        fprintf(stderr, "\rLignes restantes: %d ", j); //debug dans la console
        fflush(stderr);  
        for (int i = 0; i < largeur_image; ++i) { 

            color pixel_color = BLACK;
      
            for (int x=0; x<nbRayonParPixel; ++x){
                double u = ((double)i+randomDouble(-0.5, 0.5))/(largeur_image-1);
                double v = ((double)j+randomDouble(-0.5, 0.5))/(hauteur_image-1);

                ray r = {origin, add(coin_bas_gauche, add(multiply_scalar(horizontal, u), sub(multiply_scalar(vertical, v), origin)))}; 
                // origine = (0,0,0) et direction = coin_bas_gauche + u*horizontal + v*vertical - origine) : pour faire tout les points du viewport
            
                pixel_color = add(pixel_color, ray_color(r));
            }
            write_color(fichier, pixel_color);
        }
    }
    fclose(fichier);

    
    gettimeofday(&end_time, NULL);

    long seconds = end_time.tv_sec - start_time.tv_sec;

    fprintf(stderr, "\nFini.\n");
    fprintf(stderr, "\nTemps d'exécution : %ld min %ld sec\n", seconds / 60, seconds % 60);

	return 0;
}