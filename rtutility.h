#ifndef RTUTILITY_H
#define RTUTILITY_H

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include "ray.h"
#include "vec3.h"

// constantes

const double ratio = 16.0/10.0;

const int largeur_image = 1000;
const int hauteur_image = (int)(largeur_image / ratio);

const int nbRayonParPixel = 100;
const int nbRebondMax = 5;

#define NUM_THREADS 16



#define RED {{1, 0, 0}}
#define GREEN {{0, 1, 0}}
#define BLUE {{0, 0, 1}}
#define WHITE {{1, 1, 1}}
#define BLACK {{0, 0, 0}}
#define SKY {{0.784, 0.965, 1}}
const point3 light = {{-1, -1, -1}}; // position de la lumière -> pour "fake shadow"

const double pi = 3.1415926535897932385;

// fonctions

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

    r = sqrtf(rapport*r);
    g = sqrtf(rapport*g);
    b = sqrtf(rapport*b);

    // ecrit la valeur transposée de [0,255] de chaque composante de couleur (rgb)
    fprintf(out, "%d %d %d\n", (int)(256 * clamp(r, 0.0, 0.999)), (int)(256 * clamp(g, 0.0, 0.999)), (int)(256 * clamp(b, 0.0, 0.999)));
}


void base_ppm(FILE *out){
    fprintf(out, "P3\n%d %d\n255\n", largeur_image, hauteur_image);
}

color write_color_canva(color pixel_color){
    
    double r = pixel_color.e[0];
    double g = pixel_color.e[1];
    double b = pixel_color.e[2];
    
    double rapport = 1.0/nbRayonParPixel;

    r = sqrtf(rapport*r);
    g = sqrtf(rapport*g);
    b = sqrtf(rapport*b);

    // ecrit la valeur transposée de [0,255] de chaque composante de couleur (rgb)
    color res = {{(int)(256 * clamp(r, 0.0, 0.999)), (int)(256 * clamp(g, 0.0, 0.999)), (int)(256 * clamp(b, 0.0, 0.999))}};
    return res;
}

void canva_to_ppm(FILE *out, color* canva){
    for (int j = hauteur_image-1; j >= 0  ; j--){ 
        for (int i = 0; i < largeur_image; i++){
            fprintf(out, "%d %d %d\n", (int)canva[j*largeur_image+i].e[0], (int)canva[j*largeur_image+i].e[1], (int)canva[j*largeur_image+i].e[2]);
        }
    }
}

// direction aléatoire
vec3 random_dir(vec3 normal){
    vec3 dir;

    double u = rand()/(RAND_MAX + 1.0);
    double v = rand()/(RAND_MAX + 1.0);
    
    double theta = 2*pi*u;
    double phi = acos(2*v - 1);

    dir.e[0] = cos(theta)*sin(phi);
    dir.e[1] = sin(theta)*sin(phi);
    dir.e[2] = cos(phi);

    if (vec3_dot(dir, normal) >= 0){
        return vec3_normalize(dir);
    }
    else{
        return vec3_negate(vec3_normalize(dir));
    }
}

double randomDouble(double min, double max){
    return min + (max-min)*(rand()/(RAND_MAX + 1.0));
}

#endif