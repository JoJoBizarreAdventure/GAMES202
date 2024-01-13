#ifdef GL_ES
precision highp float;
#endif

uniform vec3 uLightDir;
uniform vec3 uCameraPos;
uniform vec3 uLightRadiance;
uniform sampler2D uGDiffuse;
uniform sampler2D uGDepth;
uniform sampler2D uGNormalWorld;
uniform sampler2D uGShadow;
uniform sampler2D uGPosWorld;

varying mat4 vWorldToScreen;
varying highp vec4 vPosWorld;

#define M_PI 3.1415926535897932384626433832795
#define TWO_PI 6.283185307
#define INV_PI 0.31830988618
#define INV_TWO_PI 0.15915494309

float Rand1(inout float p) {
  p = fract(p * .1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

vec2 Rand2(inout float p) {
  return vec2(Rand1(p), Rand1(p));
}

float InitRand(vec2 uv) {
	vec3 p3  = fract(vec3(uv.xyx) * .1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

vec3 SampleHemisphereUniform(inout float s, out float pdf) {
  vec2 uv = Rand2(s);
  float z = uv.x;
  float phi = uv.y * TWO_PI;
  float sinTheta = sqrt(1.0 - z*z);
  vec3 dir = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);
  pdf = INV_TWO_PI;
  return dir;
}

vec3 SampleHemisphereCos(inout float s, out float pdf) {
  vec2 uv = Rand2(s);
  float z = sqrt(1.0 - uv.x);
  float phi = uv.y * TWO_PI;
  float sinTheta = sqrt(uv.x);
  vec3 dir = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);
  pdf = z * INV_PI;
  return dir;
}


/** details at https://backend.orbit.dtu.dk/ws/portalfiles/portal/126824972/onb_frisvad_jgt2012_v2.pdf
*   Building an Orthonormal Basis froma 3D Unit Vector n
**/
void LocalBasis(vec3 n, out vec3 b1, out vec3 b2) {
  float sign_ = sign(n.z);
  if (n.z == 0.0) {
    sign_ = 1.0;
  }
  float a = -1.0 / (sign_ + n.z);
  float b = n.x * n.y * a;
  b1 = vec3(1.0 + sign_ * n.x * n.x * a, sign_ * b, -sign_ * n.x);
  b2 = vec3(b, sign_ + n.y * n.y * a, -n.y);
}

vec4 Project(vec4 a) {
  return a / a.w;
}

float GetDepth(vec3 posWorld) {
  float depth = (vWorldToScreen * vec4(posWorld, 1.0)).w;
  return depth;
}

/*
 * Transform point from world space to screen space([0, 1] x [0, 1])
 *
 */
vec2 GetScreenCoordinate(vec3 posWorld) {
  vec2 uv = Project(vWorldToScreen * vec4(posWorld, 1.0)).xy * 0.5 + 0.5;
  return uv;
}

float GetGBufferDepth(vec2 uv) {
  float depth = texture2D(uGDepth, uv).x;
  if (depth < 1e-2) {
    depth = 1000.0;
  }
  return depth;
}

vec3 GetGBufferNormalWorld(vec2 uv) {
  vec3 normal = texture2D(uGNormalWorld, uv).xyz;
  return normal;
}

vec3 GetGBufferPosWorld(vec2 uv) {
  vec3 posWorld = texture2D(uGPosWorld, uv).xyz;
  return posWorld;
}

float GetGBufferuShadow(vec2 uv) {
  float visibility = texture2D(uGShadow, uv).x;
  return visibility;
}

vec3 GetGBufferDiffuse(vec2 uv) {
  vec3 diffuse = texture2D(uGDiffuse, uv).xyz;
  diffuse = pow(diffuse, vec3(2.2));
  return diffuse;
}

/*
 * Evaluate diffuse bsdf value.
 *
 * wi, wo are all in world space.
 * uv is in screen space, [0, 1] x [0, 1].
 *
 */
vec3 EvalDiffuse(vec3 wi, vec2 uv) {
  vec3 albedo = GetGBufferDiffuse(uv);
  vec3 normalWorld = GetGBufferNormalWorld(uv);
  
  vec3 fr = albedo * INV_PI * max(0.0, dot(wi, normalWorld));
  return fr;
}

/*
 * Evaluate directional light with shadow map
 * uv is in screen space, [0, 1] x [0, 1].
 *
 */
vec3 EvalDirectionalLight(vec2 uv) {
  float visibility = GetGBufferuShadow(uv);

  vec3 Le = uLightRadiance * visibility;
  return Le;
}

bool RayMarch(vec3 ori, vec3 dir, out vec3 hitPos) {
  float stepLength = 0.05;
  const int maxSteps = 150;

  dir = normalize(dir);

  vec3 pos = ori;

  for(int stepCnt = 0; stepCnt < maxSteps; stepCnt++){
    vec2 uv = GetScreenCoordinate(pos);
    float depth = GetDepth(pos);
    float gBufferDepth = GetGBufferDepth(uv);

    if(depth >= gBufferDepth + 1e-4){
      hitPos = pos;
      return true;
    }

    pos += stepLength * dir;
  }

  return false;
}

vec3 EvalReflect(vec3 wi, vec3 wo, vec2 uv) {
  vec3 worldNormal = GetGBufferNormalWorld(uv);
  vec3 relfectDir = normalize(reflect(-wo, worldNormal));
  vec3 hitPos;
  if(RayMarch(vPosWorld.xyz, relfectDir, hitPos)){
      vec2 screenUV = GetScreenCoordinate(hitPos);
      return GetGBufferDiffuse(screenUV);
  }
  else{
    return vec3(0.); 
  }
}

#define SAMPLE_NUM 5

void main() {
  float s = InitRand(gl_FragCoord.xy);

  vec2 uv = GetScreenCoordinate(vPosWorld.xyz);
  vec3 wi = normalize(uLightDir);

  vec3 L = vec3(0.0);
  // dir light
  vec3 L_dir = EvalDirectionalLight(uv) * EvalDiffuse(wi, uv);

  // indir light
  vec3 L_indir = vec3(0.0);
  for(int i=0; i<SAMPLE_NUM; i++){
    float pdf = 0.0;
    // vec3 sampleDir = SampleHemisphereUniform(s, pdf);
    vec3 sampleDir = SampleHemisphereCos(s, pdf);
    
    // turn sampleDir to world space
    vec3 normal = GetGBufferNormalWorld(uv);
    vec3 b1, b2;
    LocalBasis(normal, b1, b2);
    vec3 dir = normalize(mat3(b1, b2, normal) * sampleDir);

    vec3 hitPos;
    if(RayMarch(vPosWorld.xyz, dir, hitPos)){
      vec2 hitUv = GetScreenCoordinate(hitPos);
      vec3 lightSource = EvalDirectionalLight(hitUv) * EvalDiffuse(wi, hitUv);
      L_indir += lightSource * EvalDiffuse(dir, uv) / pdf;
    }
  }
  L_indir /= float(SAMPLE_NUM);

  // Screen Space Ray Tracing 的反射测试
  // vec3 wo = normalize(uCameraPos - vPosWorld.xyz);
  // L = (GetGBufferDiffuse(uv) + EvalReflect(wi, wo, uv))/2.;

  L = L_dir + L_indir;

  vec3 color = pow(clamp(L, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
  gl_FragColor = vec4(vec3(color.rgb), 1.0);
}
