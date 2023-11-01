#ifndef HITINFO_H
#define HITINFO_H

#include "vec3.h"

typedef struct Material {
    color diffuseColor;
    color emissionColor;
    double emissionStrength;
    double reflectionStrength;
    double alpha;
    double materialIndex;
} material;

typedef struct {
    bool didHit;
    double dst;
    point3 hitPoint;
    vec3 normal;
    material mat;
} HitInfo;

#endif