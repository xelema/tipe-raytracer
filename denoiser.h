#ifndef DENOISER_H
#define DENOISER_H

#include "vec3.h"
#include "ray.h"
#include "hitinfo.h"
#include "sphere.h"
#include "rtutility.h"
#include "camera.h"

typedef struct Col_Alb_Norm{
	vec3 e[3];
} col_alb_norm;

col_alb_norm can_create (vec3 e0, vec3 e1, vec3 e2){
	col_alb_norm res;
	res.e[0] = e0;
	res.e[1] = e1;
	res.e[2] = e2;
	return res;
}

col_alb_norm add_col_alb_norm(col_alb_norm v1, col_alb_norm v2){
    col_alb_norm result;
    result.e[0] = add(v1.e[0], v2.e[0]);
    result.e[1] = add(v1.e[1], v2.e[1]);
    result.e[2] = add(v1.e[2], v2.e[2]);
    return result;
}

void denoiser(int largeur_image, int hauteur_image, color* canva, camera cam, color* albedo, color* normal){

    // initialise le device OIDN
    OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);
    oidnCommitDevice(device);

    OIDNBuffer colorBuf  = oidnNewBuffer(device, largeur_image * hauteur_image * 3 * sizeof(float));
    OIDNBuffer albedoBuf = oidnNewBuffer(device, largeur_image * hauteur_image * 3 * sizeof(float));
    OIDNBuffer normalBuf = oidnNewBuffer(device, largeur_image * hauteur_image * 3 * sizeof(float));

    float* floatImage = (float*)oidnGetBufferData(colorBuf);
    for(int i = 0; i < largeur_image * hauteur_image; i++) {
        floatImage[3*i] = (float)canva[i].e[0] / 255.0f;
        floatImage[3*i+1] = (float)canva[i].e[1] / 255.0f;
        floatImage[3*i+2] = (float)canva[i].e[2] / 255.0f;
    }

    float* floatAlbedo = (float*)oidnGetBufferData(albedoBuf);
    for(int i = 0; i < largeur_image * hauteur_image; i++) {
        floatAlbedo[3*i] = (float)albedo[i].e[0];
        floatAlbedo[3*i+1] = (float)albedo[i].e[1];
        floatAlbedo[3*i+2] = (float)albedo[i].e[2];
    }

    float* floatNormal = (float*)oidnGetBufferData(normalBuf);
    for(int i = 0; i < largeur_image * hauteur_image; i++) {
        floatNormal[3*i] = (float)normal[i].e[0];
        floatNormal[3*i+1] = (float)normal[i].e[1];
        floatNormal[3*i+2] = (float)normal[i].e[2];
    }

    // crée le filtre ("RT" est pour raytracing)
    OIDNFilter filter = oidnNewFilter(device, "RT");
    oidnSetFilterImage(filter, "color",  colorBuf, OIDN_FORMAT_FLOAT3, largeur_image, hauteur_image, 0, 0, 0); // base
    oidnSetFilterImage(filter, "albedo", albedoBuf, OIDN_FORMAT_FLOAT3, largeur_image, hauteur_image, 0, 0, 0); // albedo (couleure sans lumiere)
    oidnSetFilterImage(filter, "normal", normalBuf, OIDN_FORMAT_FLOAT3, largeur_image, hauteur_image, 0, 0, 0); // normal
    oidnSetFilterImage(filter, "output", colorBuf, OIDN_FORMAT_FLOAT3, largeur_image, hauteur_image, 0, 0, 0); // denoise
    oidnCommitFilter(filter);

    // applique le filtre
    oidnExecuteFilter(filter);

    // check les erreurs du denoise
    const char* errorMessage;
    if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE){
        printf("Error: %s\n", errorMessage);
    }

    // retransfère les valeurs dans canva
    for (int i = 0; i < largeur_image * hauteur_image; i++) {
        canva[i].e[0] = (int)(floatImage[3*i] * 255.0f);
        canva[i].e[1] = (int)(floatImage[3*i+1] * 255.0f);
        canva[i].e[2] = (int)(floatImage[3*i+2] * 255.0f);
    }

    oidnReleaseBuffer(colorBuf);
    oidnReleaseBuffer(albedoBuf);
    oidnReleaseBuffer(normalBuf);
    oidnReleaseFilter(filter);
    oidnReleaseDevice(device);
}

#endif