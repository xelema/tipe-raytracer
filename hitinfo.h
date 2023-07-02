#ifndef HITINFO_H
#define HITINFO_H

#include "vec3.h"
#include "sphere.h"

typedef struct{
    bool didHit;
    double dst;
    point3 hitPoint;
    vec3 normal;
    color mat;
} HitInfo;

#endif