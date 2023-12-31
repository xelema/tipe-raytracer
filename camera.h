#ifndef CAMERA_H
#define CAMERA_H


#include "rtutility.h"
#include "vec3.h"

// camera

typedef struct {
    point3 origin;
    vec3 horizontal;
    vec3 vertical;
    point3 coin_bas_gauche;
} camera;

double degrees_to_radians(double angle){
    return angle * 3.1415926535897932385 / 180.0;
}

camera init_camera(point3 origin, point3 target, vec3 up, double vfov, double ratio){

    camera cam;
    double theta = degrees_to_radians(vfov);
    double h = tan(theta/2);
    double hauteur_viewport = 2.0 * h;
    double largeur_viewport = ratio * hauteur_viewport;

    vec3 w = vec3_normalize(sub(origin, target));
    vec3 u = vec3_normalize(vec3_cross(up, w));
    vec3 v = vec3_cross(w, u);

    cam.origin = origin;
    cam.horizontal = multiply_scalar(u, largeur_viewport);
    cam.vertical = multiply_scalar(v, hauteur_viewport);
    cam.coin_bas_gauche = sub(cam.origin, add(divide_scalar(cam.horizontal,2), add(divide_scalar(cam.vertical, 2), w)));
    //coin_bas_gauche = origin - horizontal/2 - vertical/2 - profondeur

    return cam;
}

ray get_ray(double u, double v, camera cam, double focus_distance, double dx_ouverture, double dy_ouverture){
    ray res;

    vec3 direction = add(cam.coin_bas_gauche, add(multiply_scalar(cam.horizontal, u), sub(multiply_scalar(cam.vertical, v), cam.origin)));

    vec3 destination = add(cam.origin, multiply_scalar(direction, focus_distance));
    point3 new_origin = add(cam.origin, vec3_create(dx_ouverture, dy_ouverture, 0));

    res.origin = new_origin;
    res.dir = vec3_normalize(sub(destination, new_origin));

    return res;
}


#endif
