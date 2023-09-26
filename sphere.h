#ifndef SPHERE_H
#define SPHERE_H

#include "vec3.h"
#include "hitinfo.h"

typedef struct Sphere{
    point3 center;
    double radius;
    material mat;
} sphere;

HitInfo hit_sphere(point3 center, double radius, ray r){

    HitInfo hitInfo;
    hitInfo.didHit=false; 

    //si delta>0 alors spheres il y a

    vec3 oc = sub(r.origin, center);
    double a = vec3_dot(r.dir, r.dir);
    double b = 2.0*vec3_dot(oc, r.dir);
    double c = vec3_dot(oc, oc) - radius*radius;
    
    double discriminant = b*b - 4*a*c;

    if (discriminant > 0){
        double t1 = (-b - sqrt(discriminant))/(2*a);
        if (t1 >= 0){
            hitInfo.didHit = true;
            hitInfo.dst = t1;
            hitInfo.hitPoint = ray_at(r, t1);
            hitInfo.normal = vec3_normalize(sub(ray_at(r, t1), center));
            return hitInfo;
        }

        double t2 = (-b + sqrt(discriminant))/(2*a);
        if (t2 >= 0.001){
            hitInfo.didHit = true;
            hitInfo.dst = t2;
            hitInfo.hitPoint = ray_at(r, t2);
            hitInfo.normal = vec3_normalize(sub(ray_at(r, t2), center));
            return hitInfo;
        }
    }
    return hitInfo;
}

#endif  