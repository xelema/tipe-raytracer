#ifndef TEXTURE_H
#define TEXTURE_H

#include "mesh.h"

point3 get_barycentric_coord(triangle tri, HitInfo hitInfo){
    point3 res;
    double areaABC = vec3_dot(hitInfo.normal, vec3_cross(sub(tri.B, tri.A), sub(tri.C, tri.A)));
    double areaPBC = vec3_dot(hitInfo.normal, vec3_cross(sub(tri.B, hitInfo.hitPoint), sub(tri.C, hitInfo.hitPoint)));
    double areaPCA = vec3_dot(hitInfo.normal, vec3_cross(sub(tri.C, hitInfo.hitPoint), sub(tri.A, hitInfo.hitPoint)));

    res.e[0] = areaPBC / areaABC;
    res.e[1] = areaPCA / areaABC;
    res.e[2] = 1.0 - res.e[0] - res.e[1];

    return res;
}

color get_texture_color(triangle tri, HitInfo hitInfo, color* text_list, int texture_width, int texture_height) {
    UV uv;
    point3 bary = get_barycentric_coord(tri, hitInfo);

    // coordonnée uv du point (entre 0 et 1)
    uv.u = (bary.e[0]*tri.uvA.u + bary.e[1]*tri.uvB.u + bary.e[2]*tri.uvC.u);
    uv.v = (bary.e[0]*tri.uvA.v + bary.e[1]*tri.uvB.v + bary.e[2]*tri.uvC.v);

    // coordonnées correspondant à la texture
    int x = (int)(uv.u * (double)(texture_width));
    int y = (int)(uv.v * (double)(texture_height));

    return text_list[y * texture_width + x];
}

color* create_tex_list(const char *filename, int *width, int *height) {
    FILE *file = fopen(filename, "r");

    char format[3];
    fscanf(file, "%s", format);

    fscanf(file, "%d %d", width, height);
    int maxVal;
    fscanf(file, "%d", &maxVal);

    color* tex_list = (color *)malloc((*width)*(*height)*sizeof(color));  

    int index = 0;
    for (int i = (*height)-1; i >= 0 ; i--){
        for (int j = (*width)-1; j >= 0 ; j--){
            index = i * (*width) + j;
            tex_list[index].e[0] = 0;
            tex_list[index].e[1] = 0;
            tex_list[index].e[2] = 0;
            fscanf(file, "%lf %lf %lf", &(tex_list)[index].e[0], &tex_list[index].e[1], &tex_list[index].e[2]);
            
            tex_list[index].e[0] /= maxVal;
            tex_list[index].e[1] /= maxVal;
            tex_list[index].e[2] /= maxVal;
        }
    }
    fclose(file);
    return tex_list;
}

#endif