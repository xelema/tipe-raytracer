#ifndef VEC3_H
#define VEC3_H

#include <stdio.h>
#include <math.h>

typedef struct Vec3{
	double e[3];
} vec3;

vec3 vec3_init() {
    vec3 v;
    v.e[0] = 0;
    v.e[1] = 0;
    v.e[2] = 0;
    return v;
}

vec3 vec3_create(double e0, double e1, double e2) {
    vec3 v;
    v.e[0] = e0;
    v.e[1] = e1;
    v.e[2] = e2;
    return v;
}

double vec3_x(vec3 v) {
    return v.e[0];
}

double vec3_y(vec3 v) {
    return v.e[1];
}

double vec3_z(vec3 v) {
    return v.e[2];
}

vec3 vec3_negate(vec3 v) {
    vec3 result;
    result.e[0] = -v.e[0];
    result.e[1] = -v.e[1];
    result.e[2] = -v.e[2];
    return result;
}

double vec3_get(vec3 v, int i) {
    return v.e[i];
}

double vec3_length_squared(vec3 v) {
    return v.e[0]*v.e[0] + v.e[1]*v.e[1] + v.e[2]*v.e[2];
}

// vec3 Utility Functions

vec3 add(vec3 v1, vec3 v2){
    vec3 result;
    result.e[0] = v1.e[0] + v2.e[0];
    result.e[1] = v1.e[1] + v2.e[1];
    result.e[2] = v1.e[2] + v2.e[2];
    return result;
}

vec3 add_scalar(vec3 v, double t){
    vec3 result;
    result.e[0] = v.e[0] + t;
    result.e[1] = v.e[1] + t;
    result.e[2] = v.e[2] + t;
    return result;
}

vec3 sub(vec3 u, vec3 v){
    vec3 result;
    result.e[0] = u.e[0] - v.e[0];
    result.e[1] = u.e[1] - v.e[1];
    result.e[2] = u.e[2] - v.e[2];
    return result;
}

vec3 sub_scalar(vec3 v, double t){
    vec3 result;
    result.e[0] = v.e[0] - t;
    result.e[1] = v.e[1] - t;
    result.e[2] = v.e[2] - t;
    return result;
}

vec3 multiply(vec3 u, vec3 v) {
    vec3 result;
    result.e[0] = u.e[0] * v.e[0];
    result.e[1] = u.e[1] * v.e[1];
    result.e[2] = u.e[2] * v.e[2];
    return result;
}

vec3 multiply_scalar(vec3 v, double t){
    vec3 result;
    result.e[0] = v.e[0] * t;
    result.e[1] = v.e[1] * t;
    result.e[2] = v.e[2] * t;
    return result;
}

vec3 divide(vec3 v, double t) {
    return multiply_scalar(v, 1/t);
}

vec3 divide_scalar(vec3 v, double t) {
    vec3 result;
    result.e[0] = v.e[0] / t;
    result.e[1] = v.e[1] / t;
    result.e[2] = v.e[2] / t;
    return result;
}

double vec3_dot(vec3 u, vec3 v){
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

vec3 vec3_cross(vec3 u, vec3 v){
    vec3 result;
    result.e[0] = u.e[1] * v.e[2] - u.e[2] * v.e[1];
    result.e[1] = u.e[2] * v.e[0] - u.e[0] * v.e[2];
    result.e[2] = u.e[0] * v.e[1] - u.e[1] * v.e[0];
    return result;
}

double vec3_length(vec3 v) {
    return sqrt(vec3_dot(v, v));
}

double vec3_distance(vec3 u, vec3 v){
    return sqrt((v.e[0]-u.e[0])*(v.e[0]-u.e[0]) + (v.e[1]-u.e[1])*(v.e[1]-u.e[1]) + (v.e[2]-u.e[2])*(v.e[2]-u.e[2]));
}

vec3 vec3_normalize(vec3 v){
    return divide_scalar(v, vec3_length(v));
}

void print_vec3(vec3 v) {
    printf("%f %f %f\n", v.e[0], v.e[1], v.e[2]);
}

typedef vec3 point3;
typedef vec3 color;

#endif