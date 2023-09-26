#ifndef RTUTILITY_H
#define RTUTILITY_H

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include "ray.h"
#include "vec3.h"

// constantes

#define PI 3.1415926535897932385

#define RED {{1, 0, 0}}
#define GREEN {{0, 1, 0}}
#define BLUE {{0, 0, 1}}
#define WHITE {{1, 1, 1}}
#define BLACK {{0, 0, 0}}
#define SKY {{0.784, 0.965, 1}}
const point3 light = {{-1, -1, -1}}; // position de la lumière -> pour "fake shadow"

// fonctions

double clamp(double x, double min, double max){
    if (x<min) return min;
    if (x>max) return max;
    return x;
}

vec3 vec3_lerp(vec3 x, vec3 y, double t){
    vec3 res;
    res = add(x, multiply_scalar(sub(y, x), t));
    return res;
}

// void write_color(FILE *out, color pixel_color) {
//     double r = pixel_color.e[0];
//     double g = pixel_color.e[1];
//     double b = pixel_color.e[2];
    
//     double rapport = 1.0/nbRayonParPixel;

//     r = sqrtf(rapport*r);
//     g = sqrtf(rapport*g);
//     b = sqrtf(rapport*b);

//     // ecrit la valeur transposée de [0,255] de chaque composante de couleur (rgb)
//     fprintf(out, "%d %d %d\n", (int)(256 * clamp(r, 0.0, 0.999)), (int)(256 * clamp(g, 0.0, 0.999)), (int)(256 * clamp(b, 0.0, 0.999)));
// }


void base_ppm(FILE *out, int largeur_image, int hauteur_image){
    fprintf(out, "P3\n%d %d\n255\n", largeur_image, hauteur_image);
}

color write_color_canva(color pixel_color, int nbRayonParPixel){
    
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

void canva_to_ppm(FILE *out, color* canva, int hauteur_image, int largeur_image){
    for (int j = hauteur_image-1; j >= 0  ; j--){ 
        for (int i = 0; i < largeur_image; i++){
            fprintf(out, "%d %d %d\n", (int)canva[j*largeur_image+i].e[0], (int)canva[j*largeur_image+i].e[1], (int)canva[j*largeur_image+i].e[2]);
        }
    }
}

color rgb_to_hsl(color rgb) {
    double r = rgb.e[0];
    double g = rgb.e[1];
    double b = rgb.e[2];

    double max = (r > g) ? ((r > b) ? r : b) : ((g > b) ? g : b);
    double min = (r < g) ? ((r < b) ? r : b) : ((g < b) ? g : b);
    double h, s, l;

    l = (max + min) / 2.0;

    if (max == min) {
        h = 0.0;
        s = 0.0;
    } else {
        double d = max - min;
        s = (l < 0.5) ? (d / (max + min)) : (d / (2.0 - max - min));

        if (max == r) {
            h = (g - b) / d + ((g < b) ? 6.0 : 0.0);
        } else if (max == g) {
            h = (b - r) / d + 2.0;
        } else if (max == b) {
            h = (r - g) / d + 4.0;
        }

        h /= 6.0;
    }

    color hsl;
    hsl.e[0] = h;
    hsl.e[1] = s;
    hsl.e[2] = l;

    return hsl;
}

double hue_to_rgb(double temp1, double temp2, double hue) {
    if (hue < 0.0) {
        hue += 1.0;
    }
    if (hue > 1.0) {
        hue -= 1.0;
    }

    if (6.0 * hue < 1.0) {
        return temp1 + (temp2 - temp1) * 6.0 * hue;
    }
    if (2.0 * hue < 1.0) {
        return temp2;
    }
    if (3.0 * hue < 2.0) {
        return temp1 + (temp2 - temp1) * ((2.0 / 3.0) - hue) * 6.0;
    }

    return temp1;
}

color hsl_to_rgb(color hsl) {
    double h = hsl.e[0];
    double s = hsl.e[1];
    double l = hsl.e[2];

    double r, g, b;

    if (s == 0.0) {
        r = l;
        g = l;
        b = l;
    } else {
        double temp2 = (l < 0.5) ? (l * (1.0 + s)) : (l + s - l * s);
        double temp1 = 2.0 * l - temp2;

        r = hue_to_rgb(temp1, temp2, h + 1.0 / 3.0);
        g = hue_to_rgb(temp1, temp2, h);
        b = hue_to_rgb(temp1, temp2, h - 1.0 / 3.0);
    }

    color rgb;
    rgb.e[0] = r;
    rgb.e[1] = g;
    rgb.e[2] = b;

    return rgb;
}

// direction aléatoire
vec3 random_dir(vec3 normal){
    vec3 dir;

    double u = rand()/(RAND_MAX + 1.0);
    double v = rand()/(RAND_MAX + 1.0);
    
    double theta = 2*PI*u;
    double phi = acos(2*v - 1);

    dir.e[0] = cosf(theta)*sinf(phi);
    dir.e[1] = sinf(theta)*sinf(phi);
    dir.e[2] = cosf(phi);

    if (vec3_dot(dir, normal) >= 0){
        return vec3_normalize(dir);
    }
    else{
        return vec3_negate(vec3_normalize(dir));
    }
}

vec3 random_dir_no_norm(){
    vec3 dir;

    double u = rand()/(RAND_MAX + 1.0);
    double v = rand()/(RAND_MAX + 1.0);
    
    double theta = 2*PI*u;
    double phi = acos(2*v - 1);

    dir.e[0] = cosf(theta)*sinf(phi);
    dir.e[1] = sinf(theta)*sinf(phi);
    dir.e[2] = cosf(phi);

    return vec3_normalize(dir);
}

double randomDouble(double min, double max){
    return min + (max-min)*(rand()/(RAND_MAX + 1.0));
}

#endif