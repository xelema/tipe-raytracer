#ifndef SPHERE_H
#define SPHERE_H

#include "vec3.h"

typedef struct {
    color diffuseColor;
    color emissionColor;
    double emissionStrength;
} material;

typedef struct Sphere{
    point3 center;
    double radius;
    material mat;
} sphere;

#endif  