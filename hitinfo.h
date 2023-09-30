#ifndef HITINFO_H
#define HITINFO_H

#include "vec3.h"
#include "rtutility.h"

typedef struct {
    color diffuseColor;
    color emissionColor;
    double emissionStrength;
    double reflectionStrength;
} material;

color checker_texture_value(color c1, color c2, double scale, point3 p) {
    int x = (int)floor(1.0 / scale * p.e[0]);
    int y = (int)floor(1.0 / scale * p.e[1]);
    int z = (int)floor(1.0 / scale * p.e[2]);
    if ((x + y + z) % 2 == 0) return c1;
    else return c2;
}

typedef struct{
    bool didHit;
    double dst;
    point3 hitPoint;
    vec3 normal;
    material mat;
    double u;
    double v;
} HitInfo;

#endif