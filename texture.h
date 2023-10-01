#ifndef TEXTURE_H
#define TEXTURE_H

#include "hitinfo.h"
#include "mesh.h"

color checker_texture_value(color c1, color c2, double scale, point3 p) {
    int x = (int)floor(1.0 / scale * p.e[0]);
    int y = (int)floor(1.0 / scale * p.e[1]);
    int z = (int)floor(1.0 / scale * p.e[2]);
    if ((x + y + z) % 2 == 0) return c1;
    else return c2;
}

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

color get_texture_color(triangle tri, HitInfo hitInfo, material* mat_list, int texture_width, int texture_height) {
    UV uv;
    point3 bary = get_barycentric_coord(tri, hitInfo);

    // coordonnée uv du point (entre 0 et 1)
    uv.u = (bary.e[0]*tri.uvA.u + bary.e[1]*tri.uvB.u + bary.e[2]*tri.uvC.u);
    uv.v = (bary.e[0]*tri.uvA.v + bary.e[1]*tri.uvB.v + bary.e[2]*tri.uvC.v);

    // coordonnées correspondant à la texture
    int x = (int)(uv.u * (double)(texture_width));
    int y = (int)(uv.v * (double)(texture_height));

    return mat_list[y * texture_width + x].diffuseColor;
}

material get_material_from_matlist(triangle tri, HitInfo hitInfo, material* mat_list, int texture_width, int texture_height) {
    UV uv;
    point3 bary = get_barycentric_coord(tri, hitInfo);

    // coordonnée uv du point (entre 0 et 1)
    uv.u = (bary.e[0]*tri.uvA.u + bary.e[1]*tri.uvB.u + bary.e[2]*tri.uvC.u);
    uv.v = (bary.e[0]*tri.uvA.v + bary.e[1]*tri.uvB.v + bary.e[2]*tri.uvC.v);

    // coordonnées correspondant à la texture
    int x = (int)(uv.u * (double)(texture_width));
    int y = (int)(uv.v * (double)(texture_height));

    material res;

    res.diffuseColor = mat_list[y * texture_width + x].diffuseColor;
    res.alpha = mat_list[y * texture_width + x].alpha;

    return res;
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
    for (int i = (*height)-1; i >= 0; i--){
        for (int j = 0; j < (*width); j++){
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

material* create_mat_list(const char *texfile, const char *alphafile, int *width, int *height) {
    FILE *tex_file = fopen(texfile, "r");
    FILE *alpha_file = fopen(alphafile, "r");


    char format[3];
    fscanf(tex_file, "%s", format);
    fscanf(alpha_file, "%s", format);

    fscanf(tex_file, "%d %d", width, height);
    fscanf(alpha_file, "%d %d", width, height);
    int maxVal, max_alphaVal;
    fscanf(tex_file, "%d", &maxVal);
    fscanf(alpha_file, "%d", &max_alphaVal);

    material* mat_list = (material*)malloc((*width)*(*height)*sizeof(material));  

    int index = 0;
    for (int i = (*height)-1; i >= 0; i--){
        for (int j = 0; j < (*width); j++){
            index = i * (*width) + j;
            mat_list[index].diffuseColor.e[0] = 0;
            mat_list[index].diffuseColor.e[1] = 0;
            mat_list[index].diffuseColor.e[2] = 0;
            fscanf(tex_file, "%lf %lf %lf", &(mat_list)[index].diffuseColor.e[0], &mat_list[index].diffuseColor.e[1], &mat_list[index].diffuseColor.e[2]);
            fscanf(alpha_file, "%lf %*lf %*lf", &(mat_list)[index].alpha);

            mat_list[index].diffuseColor.e[0] /= maxVal;
            mat_list[index].diffuseColor.e[1] /= maxVal;
            mat_list[index].diffuseColor.e[2] /= maxVal;

            mat_list[index].alpha /= max_alphaVal;
            // printf("mat_list[%d].alpha = %lf\n", index, mat_list[index].alpha);
        }
    }
    fclose(tex_file);
    return mat_list;
}

#endif