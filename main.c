#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

#include "vec3.h"
#include "ray.h"
#include "hitinfo.h"
#include "sphere.h"

const double ratio = 16.0 / 9.0;
const int largeur_image = 1000;
const int hauteur_image = (int)(largeur_image / ratio);

void write_color(FILE *out, color pixel_color){
    // ecrit la valeur transposée de [0,255] de chaque composante de couleur (rgb)
    fprintf(out, "%d %d %d\n", (int)(255 * pixel_color.e[0]), (int)(255 * pixel_color.e[1]), (int)(255 * pixel_color.e[2]));
}

void base_ppm(){
    printf("P3\n%d %d\n255\n", largeur_image, hauteur_image);
}

const color red = {{1, 0, 0}};
const color green = {{0, 1, 0}};
const color blue = {{0, 0, 1}};
const color white = {{1,1,1}};
const color black = {{0,0,0}};
const point3 light = {{-1, -1, -1}}; // position de la lumière

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

    sphere sphere_list[3] = {
    {{{-0.2,-0.4,-1}}, 0.5, red},      // boule rouge en bas
    {{{0,-999.8,-1}}, 999.5, green},   // sol (grosse sphere)
    {{{1, 0.5, -4}}, 1, blue},         // boule bleu en fond
    };

    HitInfo closestHit;
    closestHit.didHit=false;
    closestHit.dst=INFINITY; // rien touché pour l'instant

    for(int i=0; i < 3 /*nbSpheres*/ ; i++){
        sphere s = sphere_list[i];
        HitInfo hitInfo =  hit_sphere(s.center, s.radius, r);

        if (hitInfo.didHit && hitInfo.dst < closestHit.dst){
            closestHit = hitInfo;
            closestHit.mat = s.mat;
        }
    }
    return closestHit;
}

color ray_color(ray r) {
    HitInfo t = closest_hit(r);
    if (t.didHit == true){
        return t.mat;
    }
    else{
        return black;
    }
}

int main(){

    // camera

    double hauteur_viewport = 2.0;
    double largeur_viewport = ratio*hauteur_viewport;
    double focal_length = 1.0;

    point3 origin = {{0, 0, 0}};
    vec3 horizontal = {{largeur_viewport, 0, 0}};
    vec3 vertical = {{0, hauteur_viewport, 0}};
    vec3 focal_length_vec = {{0, 0, focal_length}};
    vec3 coin_bas_gauche = sub(origin, add(divide(horizontal,2), add(divide(vertical, 2), focal_length_vec))); 
    //coin_bas_gauche = origin - horizontal/2 - vertical/2 - profondeur

    
    // render
    base_ppm();
    
    for (int j = hauteur_image - 1; j >= 0; j--){
        for (int i = 0; i < largeur_image; i++){

            double u = (double)i/(largeur_image-1);
            double v = (double)j/(hauteur_image-1);

            ray r = {origin, add(coin_bas_gauche, add(multiply_scalar(horizontal, u), sub(multiply_scalar(vertical, v), origin)))}; 
            // origine = (0,0,0) et direction = coin_bas_gauche + u*horizontal + v*vertical - origine) 
            // pour faire tout les pixels du viewport
            
            color pixel_color = ray_color(r);
            write_color(stdout, pixel_color);
        }
    }

	return 0;
}
