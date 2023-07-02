#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include "vec3.h"
#include "ray.h"

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

///////////////////////////////////////////////////////////////////////////////////////////////


double hit_sphere(point3 center, double radius, ray r){
    double a = vec3_dot(r.dir, r.dir);
    double b = 2.0*vec3_dot(sub(r.origin, center), r.dir);
    double c = vec3_dot(sub(r.origin, center), sub(r.origin, center)) - radius*radius;
    double discriminant = b*b - 4*a*c;
    if (discriminant < 0){
        return -1.0;
    }
    else{ //si delta >= 0 alors il y a la sphere
        return (-b - sqrt(discriminant) ) / (2.0*a);
    }
}

color ray_color(ray r) {
    color black = {{0.0, 0.0, 0.0}};
    color blue = {{0.0, 0.0, 1.0}};

    point3 center = {{0,0,-1}};
    double d = hit_sphere(center, 0.5, r);
    if (d > 0.0){
        vec3 N = vec3_normalize(sub(ray_at(r, d), center));
        vec3 resN = add_scalar(multiply_scalar(N, 0.5), 0.5);
        // permet d'avoir des valeurs entre 0 et 1 et pas -1 et 1
        return resN;
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
