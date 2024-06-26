#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>
#include <random>
#include "vec.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

const int resolution = 128;

Vec2f Hammersley(uint32_t i, uint32_t N)
{ // 0-1
    uint32_t bits = (i << 16u) | (i >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    float rdi = float(bits) * 2.3283064365386963e-10;
    return {float(i) / float(N), rdi};
}

/** details at https://backend.orbit.dtu.dk/ws/portalfiles/portal/126824972/onb_frisvad_jgt2012_v2.pdf
 *   Building an Orthonormal Basis froma 3D Unit Vector n
 **/
void LocalBasis(Vec3f n, Vec3f &b1, Vec3f &b2)
{
    float sign_ = n.z < 0.0 ? -1.0 : 1.0;
    float a = -1.0 / (sign_ + n.z);
    float b = n.x * n.y * a;
    b1 = Vec3f(1.0 + sign_ * n.x * n.x * a, sign_ * b, -sign_ * n.x);
    b2 = Vec3f(b, sign_ + n.y * n.y * a, -n.y);
}

Vec3f ImportanceSampleGGX(Vec2f Xi, Vec3f N, float roughness)
{
    float a = roughness * roughness;

    // TODO: in spherical space - Bonus 1
    float theta_m = atan2(a * sqrt(Xi.x), sqrt(1 - Xi.x));
    float phi_h = 2 * PI * Xi.y;

    // TODO: from spherical space to cartesian space - Bonus 1
    float sinTheta = sin(theta_m);
    Vec3f H = {
        cos(phi_h) * sinTheta,
        sin(phi_h) * sinTheta,
        cos(theta_m)};

    // TODO: tangent coordinates - Bonus 1
    Vec3f b1,b2;
    LocalBasis(N, b1, b2);

    // TODO: transform H to tangent space - Bonus 1
    Vec3f tangentH = b1 * H.x + b2 * H.y + N * H.z;

    return normalize(tangentH);
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    // TODO: To calculate Schlick G1 here - Bonus 1
    float a = roughness;
    float k = (a * a) / 2.0f;

    float nom = NdotV;
    float denom = NdotV * (1.0f - k) + k;

    return nom / denom;
}

float GeometrySmith(float roughness, float NoV, float NoL)
{
    float ggx2 = GeometrySchlickGGX(NoV, roughness);
    float ggx1 = GeometrySchlickGGX(NoL, roughness);

    return ggx1 * ggx2;
}

#define SPLIT_SUM 1

Vec3f IntegrateBRDF(Vec3f V, float roughness)
{
    const int sample_count = 1024;
#if !SPLIT_SUM
    float sum = 0.0;
#else
    float weightSum = 0.0;
    float LiSum = 0.0;
#endif
    Vec3f N = Vec3f(0.0, 0.0, 1.0);
    for (int i = 0; i < sample_count; i++)
    {
        Vec2f Xi = Hammersley(i, sample_count);
        Vec3f H = ImportanceSampleGGX(Xi, N, roughness);
        Vec3f L = normalize(H * 2.0f * dot(V, H) - V);

        float NoL = std::max(L.z, 0.0f);
        float NoH = std::max(H.z, 0.0f);
        float VoH = std::max(dot(V, H), 0.0f);
        float NoV = std::max(dot(N, V), 0.0f);

        // TODO: To calculate (fr * ni) / p_o here - Bonus 1
        float Li = 1;
        float G = GeometrySmith(roughness, NoV, NoL);
        
        float weight = VoH * G / NoV / NoH;
#if !SPLIT_SUM
        sum += weight * Li;
#else
        // Split Sum - Bonus 2
        weightSum += weight;
        LiSum += Li;
#endif
    }
#if !SPLIT_SUM
    return Vec3f(sum / sample_count);
#else
    return Vec3f(weightSum / sample_count * LiSum / sample_count);
#endif
}

int main()
{
    uint8_t data[resolution * resolution * 3];
    float step = 1.0 / resolution;
    for (int i = 0; i < resolution; i++)
    {
        for (int j = 0; j < resolution; j++)
        {
            float roughness = step * (static_cast<float>(i) + 0.5f);
            float NdotV = step * (static_cast<float>(j) + 0.5f);
            Vec3f V = Vec3f(std::sqrt(1.f - NdotV * NdotV), 0.f, NdotV);

            Vec3f irr = IntegrateBRDF(V, roughness);

            data[(i * resolution + j) * 3 + 0] = uint8_t(irr.x * 255.0);
            data[(i * resolution + j) * 3 + 1] = uint8_t(irr.y * 255.0);
            data[(i * resolution + j) * 3 + 2] = uint8_t(irr.z * 255.0);
        }
    }
    stbi_flip_vertically_on_write(true);
    stbi_write_png("GGX_E_LUT.png", resolution, resolution, 3, data, resolution * 3);

    std::cout << "Finished precomputed!" << std::endl;
    return 0;
}