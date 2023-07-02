#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

#include "vec3.h"
#include "ray.h"
#include "hitinfo.h"
#include "sphere.h"
#include "rtutility.h"


double clamp(double x, double min, double max){
    if (x<min) return min;
    if (x>max) return max;
    return x;
}

void write_color(FILE *out, color pixel_color) {
    double r = pixel_color.e[0];
    double g = pixel_color.e[1];
    double b = pixel_color.e[2];
    
    double rapport = 1.0/nbRayonParPixel;

    r = r*rapport;
    g = g*rapport;
    b = b*rapport;

    // ecrit la valeur transposée de [0,255] de chaque composante de couleur (rgb)
    fprintf(out, "%d %d %d\n", (int)(256 * clamp(r, 0.0, 0.999)), (int)(256 * clamp(g, 0.0, 0.999)), (int)(256 * clamp(b, 0.0, 0.999)));
}


void base_ppm(){
    printf("P3\n%d %d\n255\n", largeur_image, hauteur_image);
}

const sphere sphere_list[4] = {
    {{{-0.2,-0.4,-1}}, 0.5, {{{1, 0.384, 0.384}}, {{0.0, 0.0, 0.0}}, 0.0}},       
    // sphere rouge : couleure rouge, emission noire, force d'emission 0
    {{{0,-299.8,-1}}, 299.5, {{{0.416, 0.949, 0.298}}, {{0.0, 0.0, 0.0}}, 0.0}},  
    // sol (sphere verte) : couleure verte, emission noire, force d'emission 0
    {{{1, 0.5, -4}}, 1, {{{0.843, 0.118, 0.99}}, {{0.0, 0.0, 0.0}}, 0.0}},           
    // sphere violette : couleure violette, emission noire, force d'emission 0
    {{{40, 20, -40}}, 20.0, {{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}, 8.0}},                
    // LUMIERE : couleure noire, emission blanche, force d'emission 8
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
    int nbSpheres = 4;

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

double random_value_sphere(){
    double theta = 2*pi*(double)rand()/RAND_MAX; 
    // rand()/RAND_MAX = valeur aléatoire entre 0 et 1
    double rho = sqrt(-2*log((double)rand()/RAND_MAX));
    return rho * cos(theta);
}

vec3 random_direction(){
    double x = random_value_sphere();
    double y = random_value_sphere();
    double z = random_value_sphere();
    point3 point_in_sphere = {{x, y, z}};
    return vec3_normalize(point_in_sphere);
}

vec3 bon_sens(vec3 normal){
    vec3 dir = random_direction();
    if (vec3_dot(dir, normal) >= 0){
        return dir;
    }
    else{
        return vec3_negate(dir);
    }
}

point3 tracer(ray r){

    color incomingLight = black;
    color rayColor = white;

    for (int i = 0; i<nbRebondMax; i++){

        HitInfo hitInfo = closest_hit(r);

        if (hitInfo.didHit){
            r.origin = hitInfo.hitPoint;
            r.dir = bon_sens(hitInfo.normal);

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

color ray_color(ray r) {
    color t = tracer(r);
    return t;
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
    
    for (int j = hauteur_image - 1; j >= 0  ; --j) { 
        for (int i = 0; i < largeur_image; ++i) { 

            color pixel_color = black;
      
            for (int x=0; x<nbRayonParPixel; ++x){
                double u = (double)i/(largeur_image-1);
                double v = (double)j/(hauteur_image-1);

                ray r = {origin, add(coin_bas_gauche, add(multiply_scalar(horizontal, u), sub(multiply_scalar(vertical, v), origin)))}; 
                // origine = (0,0,0) et direction = coin_bas_gauche + u*horizontal + v*vertical - origine) : pour faire tout les points du viewport
            
                pixel_color = add(pixel_color, ray_color(r));
            }
            
            write_color(stdout, pixel_color);
        }
    }


	return 0;
}
