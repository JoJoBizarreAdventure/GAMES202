#include "denoiser.h"

Denoiser::Denoiser() : m_useTemportal(false) {}

void Denoiser::Reprojection(const FrameInfo &frameInfo) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    Matrix4x4 preWorldToScreen =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 1];
    // Matrix4x4 preWorldToCamera =
    //     m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 2];
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Reproject
            m_valid(x, y) = false;
            m_misc(x, y) = Float3(0.f);

            int curObjectId = frameInfo.m_id(x, y);
            if (curObjectId == -1)
                continue;

            Matrix4x4 curLocalToWorld = frameInfo.m_matrix[curObjectId];
            Matrix4x4 preLocalToWorld = m_preFrameInfo.m_matrix[curObjectId];

            Float3 curWorldPos = frameInfo.m_position(x, y);
            Matrix4x4 curWorldToLocal = Inverse(curLocalToWorld);

            Float3 localPos = curWorldToLocal(curWorldPos, Float3::EType::Point);
            Float3 preWorldPos = preLocalToWorld(localPos, Float3::EType::Point);
            Float3 preScreenPos = preWorldToScreen(preWorldPos, Float3::EType::Point);

            int preX = preScreenPos.x, preY = preScreenPos.y;
            if (preX < 0 || preX >= width || preY < 0 || preY >= height)
                continue;

            int preObjectId = m_preFrameInfo.m_id(preX, preY);
            if (curObjectId != preObjectId)
                continue;

            m_valid(x, y) = true;
            m_misc(x, y) = m_accColor(preX, preY);
        }
    }
    std::swap(m_misc, m_accColor);
}

void Denoiser::TemporalAccumulation(const Buffer2D<Float3> &curFilteredColor) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    int kernelRadius = 3;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Temporal clamp
            Float3 color = m_accColor(x, y);
            // TODO: Exponential moving average
            float alpha = 1.0f;
            if (m_valid(x, y)) {
                alpha = m_alpha;

                int xFrom = std::max(0, x - kernelRadius),
                    xTo = std::min(width - 1, x + kernelRadius);
                int yFrom = std::max(0, y - kernelRadius),
                    yTo = std::min(height - 1, y + kernelRadius);

                int num = (xTo - xFrom + 1) * (yTo - yFrom + 1);
                Float3 colorSum(0);
                Float3 colorSqureSum(0);
                for (int v = yFrom; v <= yTo; v++) {
                    for (int u = xFrom; u <= xTo; u++) {
                        auto sampleColor = curFilteredColor(u, v);
                        colorSum += sampleColor;
                        colorSqureSum += sampleColor * sampleColor;
                    }
                }
                Float3 mu = colorSum / num;
                Float3 sigma = SafeSqrt(colorSqureSum / num - mu * mu);

                Float3 lowerLimit = mu - sigma * m_colorBoxK,
                       higherLimit = mu + sigma * m_colorBoxK;

                color = Clamp(color, lowerLimit, higherLimit);
            }

            m_misc(x, y) = Lerp(color, curFilteredColor(x, y), alpha);
        }
    }
    std::swap(m_misc, m_accColor);
}

Buffer2D<Float3> Denoiser::Filter(const FrameInfo &frameInfo) {
    int height = frameInfo.m_beauty.m_height;
    int width = frameInfo.m_beauty.m_width;
    Buffer2D<Float3> filteredImage = CreateBuffer2D<Float3>(width, height);
    int kernelRadius = 16;

    float coord_denominator = 2.0f * m_sigmaCoord * m_sigmaCoord;
    float color_denominator = 2.0f * m_sigmaColor * m_sigmaColor;
    float normal_denominator = 2.0f * m_sigmaNormal * m_sigmaNormal;
    float plane_denominator = 2.0f * m_sigmaPlane * m_sigmaPlane;

#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Joint bilateral filter
            // filteredImage(x, y) = frameInfo.m_beauty(x, y);

            int y_from = std::max(0, y - kernelRadius),
                y_to = std::min(height - 1, y + kernelRadius);
            int x_from = std::max(0, x - kernelRadius),
                x_to = std::min(width - 1, x + kernelRadius);

            double sum_of_weights = 0;
            Float3 sum_of_weighted_values(0);

            Float3 center_color = frameInfo.m_beauty(x, y);
            Float3 center_normal = frameInfo.m_normal(x, y);
            float center_normal_len = Length(center_normal);
            Float3 center_pos = frameInfo.m_position(x, y);

            for (int v = y_from; v <= y_to; v++) {
                for (int u = x_from; u <= x_to; u++) {
                    Float3 sample_color = frameInfo.m_beauty(u, v);
                    Float3 sample_normal = frameInfo.m_normal(u, v);
                    float sample_normal_len = Length(sample_normal);
                    Float3 sample_pos = frameInfo.m_position(u, v);

                    float coord_squreDistance = SqrLength(sample_pos - center_pos);
                    float J_coord = coord_squreDistance / coord_denominator;

                    float color_squreDistance = SqrDistance(center_color, sample_color);
                    float J_color = color_squreDistance / color_denominator;

                    float D_normal = SafeAcos(Dot(center_normal, sample_normal));
                    float J_normal = D_normal * D_normal / normal_denominator;

                    float D_plane = J_coord > 0 ? Dot(center_normal,
                                                      Normalize(sample_pos - center_pos))
                                                : 0.0;
                    float J_plane = D_plane * D_plane / plane_denominator;

                    double J = std::exp(-J_coord - J_color - J_normal - J_plane);

                    sum_of_weights += J;
                    sum_of_weighted_values += frameInfo.m_beauty(u, v) * J;
                }
            }
            if (sum_of_weights == 0.0) {
                filteredImage(x, y) = frameInfo.m_beauty(x, y);
            } else
                filteredImage(x, y) = sum_of_weighted_values / sum_of_weights;
        }
    }
    return filteredImage;
}

Buffer2D<Float3> Denoiser::ATrousWaveletFilter(const FrameInfo &frameInfo) {
    int height = frameInfo.m_beauty.m_height;
    int width = frameInfo.m_beauty.m_width;
    Buffer2D<Float3> filteredImage = CreateBuffer2D<Float3>(width, height);
    // int kernelRadius = 16;
    int filterTimes = 5; // kernelRadius = 16 ==> 33 * 33  5 ==> 65 * 65

    float coord_denominator = 2.0f * m_sigmaCoord * m_sigmaCoord;
    float color_denominator = 2.0f * m_sigmaColor * m_sigmaColor;
    float normal_denominator = 2.0f * m_sigmaNormal * m_sigmaNormal;
    float plane_denominator = 2.0f * m_sigmaPlane * m_sigmaPlane;

#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double sum_of_weights = 0;
            Float3 sum_of_weighted_values(0);

            Float3 center_color = frameInfo.m_beauty(x, y);
            Float3 center_normal = frameInfo.m_normal(x, y);
            float center_normal_len = Length(center_normal);
            Float3 center_pos = frameInfo.m_position(x, y);

            for (int f = 0; f <= filterTimes; f++) {
                int step = 1 << f;
                int kernelRadius = step * 2;

                int y_from = y - kernelRadius, y_to = y + kernelRadius;
                int x_from = x - kernelRadius, x_to = x + kernelRadius;

                for (int v = y_from; v <= y_to; v += step) {
                    if (v < 0 || v >= height)
                        continue;

                    for (int u = x_from; u <= x_to; u += step) {
                        if (u < 0 || u >= width)
                            continue;

                        Float3 sample_color = frameInfo.m_beauty(u, v);
                        Float3 sample_normal = frameInfo.m_normal(u, v);
                        float sample_normal_len = Length(sample_normal);
                        Float3 sample_pos = frameInfo.m_position(u, v);

                        float coord_squreDistance = SqrLength(sample_pos - center_pos);
                        float J_coord = coord_squreDistance / coord_denominator;

                        float color_squreDistance =
                            SqrDistance(center_color, sample_color);
                        float J_color = color_squreDistance / color_denominator;

                        float D_normal = SafeAcos(Dot(center_normal, sample_normal));
                        float J_normal = D_normal * D_normal / normal_denominator;

                        float D_plane =
                            J_coord > 0
                                ? Dot(center_normal, Normalize(sample_pos - center_pos))
                                : 0.0;
                        float J_plane = D_plane * D_plane / plane_denominator;

                        double J = std::exp(-J_coord - J_color - J_normal - J_plane);

                        sum_of_weights += J;
                        sum_of_weighted_values += frameInfo.m_beauty(u, v) * J;
                    }
                }
            }

            if (sum_of_weights == 0.0) {
                filteredImage(x, y) = frameInfo.m_beauty(x, y);
            } else
                filteredImage(x, y) = sum_of_weighted_values / sum_of_weights;
        }
    }

    return filteredImage;
}

void Denoiser::Init(const FrameInfo &frameInfo, const Buffer2D<Float3> &filteredColor) {
    m_accColor.Copy(filteredColor);
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    m_misc = CreateBuffer2D<Float3>(width, height);
    m_valid = CreateBuffer2D<bool>(width, height);
}

void Denoiser::Maintain(const FrameInfo &frameInfo) { m_preFrameInfo = frameInfo; }

Buffer2D<Float3> Denoiser::ProcessFrame(const FrameInfo &frameInfo) {
    // Filter current frame
    Buffer2D<Float3> filteredColor;
   
#ifdef USE_A_TROUS_WAVELET_FILTER
    filteredColor = ATrousWaveletFilter(frameInfo);
#else
    filteredColor = Filter(frameInfo);
#endif
    // Reproject previous frame color to current
    if (m_useTemportal) {
        Reprojection(frameInfo);
        TemporalAccumulation(filteredColor);
    } else {
        Init(frameInfo, filteredColor);
    }

    // Maintain
    Maintain(frameInfo);
    if (!m_useTemportal) {
        m_useTemportal = true;
    }
    return m_accColor;
}
