#ifndef RAY_H
#define RAY_H

#include "vec3.h"

typedef struct Ray{
    point3 origin;
    vec3 dir;
} ray;

ray create_ray(point3 origin, vec3 direction){
    ray r;
    r.origin = origin;
    r.dir = direction;
    return r;
}

point3 ray_origin(ray r){
    return r.origin;
}

vec3 ray_direction(ray r){
    return r.dir;
}

point3 ray_at(ray r, double t){
    // P(t) = A + tB avec t la longueur, A l'origine et B la direction
    return add(r.origin, multiply_scalar(r.dir, t));
}

#endif