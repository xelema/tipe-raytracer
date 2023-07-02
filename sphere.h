#ifndef SPHERE_H
#define SPHERE_H

#include "vec3.h"

typedef struct {
    color color;
    color emissionColor;
    double emissionStrenght;
} material;

typedef struct Sphere{
    point3 center;
    double radius;
    color mat;
} sphere;

const double pi = 3.1415926535897932385;

#endif  