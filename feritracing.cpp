//.......................................| 1 |.......................................
/*
-- First of 3 ray tracing implementations --

# Basic ray tracing implementation in C++ on Windows by FECORO.
# The Win32 library has been used for the graphical interface.
# The OpenMP library has been used for parallelization.

# This implementation features a basic scene with spheres and lights.
# It implements a basic ray tracing algorithm with diffuse shading.

-> Is possible to change the scene based on the already gave definitions, 
-> they could be expanded too and see how it works.

# To compile on Windows with MinGW64:
# g++ -o feritracer.exe feritracer.cpp -std=c++11 -lgdi32 -mwindows -lstdc++ -pthread

General structure:
- clamp: Limits a value between a minimum and maximum.
- randomFloat: Generates a random floating-point number.
- degreeToRadians: Converts degrees to radians.
- Vector3: Class for 3D vectors.
- Ray: Class for rays.
- AABB: Class for bounding boxes.
- Texture: Base class for textures.
- SolidColor: Class for solid color textures.
- Material: Class for materials.
- HitRecord: Class for storing intersection information.
- Hittable: Base class for hittable objects.
- Sphere: Class for spheres.
- Scene: Class for the scene.
- Light: Class for lights.
- Camera: Class for the camera.
- renderRowRange: Function for rendering a range of rows.
- render: Function for rendering the scene.
- Win32MainWindow: Class for the Win32 main window.
- Win32Renderer: Class for Win32 rendering.
- Win32Timer: Class for measuring time in Win32.
- Win32TimerScope: Class for measuring time within a scope in Win32.

Made by Felipe Alexander Correa Rodríguez.
*/

//.......................................| 2 |.......................................
//............... - Head stack - ...............
#include <windows.h>
#include <cmath>
#include <vector>
#include <memory>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <thread>
#include <mutex>
#include <tchar.h>

//............... - INITS - ...............
// ...constants
const float PI = 3.14159265358979323846f;
const int MAX_DEPTH = 50;
const float EPSILON = 1e-4f;

// ...utilitary
float clamp(float value, float minValue, float maxValue) {
    return value < minValue ? minValue : (value > maxValue ? maxValue : value);
}

float randomFloat() {
    return static_cast<float>(rand()) / RAND_MAX;
}

float degreeToRadians(float degrees) {
    return degrees * PI / 180.0f;
}

// ...vector3 (self made)
class Vector3 {
public:
    float x, y, z;

    Vector3(float x = 0, float y = 0, float z = 0)
        : x(x), y(y), z(z) {}

// .overloads
    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }
    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }
    Vector3 operator-() const {
        return Vector3(-x, -y, -z);
    }
    Vector3 operator*(float scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }
    Vector3 operator*(const Vector3& other) const {
        return Vector3(x * other.x, y * other.y, z * other.z);
    }
    Vector3 operator/(float scalar) const {
        return Vector3(x / scalar, y / scalar, z / scalar);
    }
    Vector3& operator+=(const Vector3& other) {
        x += other.x; y += other.y; z += other.z; return *this;
    }
    Vector3& operator*=(const Vector3& other) {
        x *= other.x; y *= other.y; z *= other.z; return *this;
    }
    Vector3& operator/=(float scalar) {
        x /= scalar; y /= scalar; z /= scalar; return *this;
    }

    // ..accessing components by index
    float& operator[](int index) {
        if (index == 0) return x;
        else if (index == 1) return y;
        else return z;
    }
    float operator[](int index) const {
        if (index == 0) return x;
        else if (index == 1) return y;
        else return z;
    }

    // ..vector operations
    float dot(const Vector3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    Vector3 cross(const Vector3& other) const {
        return Vector3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x);
    }
    float length() const {
        return sqrtf(x * x + y * y + z * z);
    }
    Vector3 normalize() const {
        float len = length(); return len != 0 ? *this / len : *this;
    }
};

// ... ray class
class Ray {
public:
    Vector3 origin, direction;

    Ray()
        : origin(Vector3()), direction(Vector3(1, 0, 0)) {}

    Ray(const Vector3& origin, const Vector3& direction)
        : origin(origin), direction(direction.normalize()) {}

    Vector3 pointAtParameter(float t) const {
        return origin + direction * t;
    }
};

// ...AABB
class AABB {
public:
    Vector3 min, max;

    AABB() {}
    AABB(const Vector3& a, const Vector3& b) : min(a), max(b) {}

    bool hit(const Ray& ray, float tmin, float tmax) const {
        for (int a = 0; a < 3; a++) {
            float invD = 1.0f / ray.direction[a];
            float t0 = (min[a] - ray.origin[a]) * invD;
            float t1 = (max[a] - ray.origin[a]) * invD;
            if (invD < 0.0f) std::swap(t0, t1);
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin) return false;
        }
        return true;
    }
};

// ...forward declarations
class Scene;
class Material;
class Light;
class Texture;

// ...texture
class Texture {
public:
    virtual Vector3 value(float u, float v, const Vector3& p) const = 0;
    virtual ~Texture() {}
};

// ...solid color
class SolidColor : public Texture {
public:
    Vector3 colorValue;

    SolidColor() {}
    SolidColor(Vector3 c) : colorValue(c) {}

    virtual Vector3 value(float u, float v, const Vector3& p) const {
        return colorValue;
    }
};

// ...checker texture
class CheckerTexture : public Texture {
public:
    std::shared_ptr<Texture> odd;
    std::shared_ptr<Texture> even;

    CheckerTexture() {}
    CheckerTexture(std::shared_ptr<Texture> t0, std::shared_ptr<Texture> t1) : even(t0), odd(t1) {}

    virtual Vector3 value(float u, float v, const Vector3& p) const {
        float sines = sinf(10 * p.x) * sinf(10 * p.y) * sinf(10 * p.z);
        if (sines < 0)
            return odd->value(u, v, p);
        else
            return even->value(u, v, p);
    }
};

// ...material base
class Material {
public:
    virtual bool scatter(const Ray& rayIn, const Vector3& intersection, const Vector3& normal,
                         Vector3& attenuation, Ray& scattered) const = 0;
    virtual Vector3 emitted(float u, float v, const Vector3& p) const {
        return Vector3(0, 0, 0);
    }
    virtual ~Material() {}
};

// ...lambertian material (basically for diffuse shading)
class Lambertian : public Material {
public:
    std::shared_ptr<Texture> albedo;

    Lambertian(const Vector3& color)
        : albedo(std::make_shared<SolidColor>(color)) {}

    Lambertian(std::shared_ptr<Texture> texture)
        : albedo(texture) {}

    virtual bool scatter(const Ray& rayIn, const Vector3& intersection, const Vector3& normal,
                         Vector3& attenuation, Ray& scattered) const {
        Vector3 scatterDirection = normal + randomUnitVector();

        // ..catching degenerate scatter direction
        if (scatterDirection.length() < EPSILON)
            scatterDirection = normal;

        scattered = Ray(intersection + normal * EPSILON, scatterDirection);
        attenuation = albedo->value(0, 0, intersection);
        return true;
    }

private:
    Vector3 randomUnitVector() const {
        float a = randomFloat() * 2.0f * PI;
        float z = randomFloat() * 2.0f - 1.0f;
        float r = sqrtf(1 - z * z);
        return Vector3(r * cosf(a), r * sinf(a), z);
    }
};

// ..metal material
class Metal : public Material {
public:
    Vector3 albedo;
    float fuzz;

    Metal(const Vector3& albedo, float fuzz)
        : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    virtual bool scatter(const Ray& rayIn, const Vector3& intersection, const Vector3& normal,
                         Vector3& attenuation, Ray& scattered) const {
        Vector3 reflected = reflect(rayIn.direction.normalize(), normal);
        scattered = Ray(intersection + normal * EPSILON, reflected + randomInUnitSphere() * fuzz);
        attenuation = albedo;
        return (scattered.direction.dot(normal) > 0);
    }

private:
    Vector3 reflect(const Vector3& v, const Vector3& n) const {
        return v - n * 2.0f * v.dot(n);
    }

    Vector3 randomInUnitSphere() const {
        Vector3 p;
        do {
            p = Vector3(randomFloat(), randomFloat(), randomFloat()) * 2.0f - Vector3(1, 1, 1);
        } while (p.length() >= 1.0f);
        return p;
    }
};

// dielectric material (glass)
/*
It works so that when a ray hits the glass, it can either be reflected or refracted.
The probability of reflection increases as the angle of incidence increases.
*/
class Dielectric : public Material {
public:
    float refIdx;

    Dielectric(float refIdx)
        : refIdx(refIdx) {}

    virtual bool scatter(const Ray& rayIn, const Vector3& intersection, const Vector3& normal,
                         Vector3& attenuation, Ray& scattered) const {
        attenuation = Vector3(1.0f, 1.0f, 1.0f);
        float etaiOverEtat;
        Vector3 unitDirection = rayIn.direction.normalize();
        Vector3 outwardNormal;
        float cosTheta;
        if (unitDirection.dot(normal) > 0) {
            outwardNormal = -normal;
            etaiOverEtat = refIdx;
            cosTheta = refIdx * unitDirection.dot(normal) / unitDirection.length();
        } else {
            outwardNormal = normal;
            etaiOverEtat = 1.0f / refIdx;
            cosTheta = -unitDirection.dot(normal) / unitDirection.length();
        }

        Vector3 refracted;
        float reflectProb;
        if (refract(unitDirection, outwardNormal, etaiOverEtat, refracted))
            reflectProb = schlick(cosTheta, refIdx);
        else
            reflectProb = 1.0f;

        if (randomFloat() < reflectProb) {
            Vector3 reflected = reflect(unitDirection, normal);
            scattered = Ray(intersection + normal * EPSILON, reflected);
        } else {
            scattered = Ray(intersection - outwardNormal * EPSILON, refracted);
        }
        return true;
    }

private:
    Vector3 reflect(const Vector3& v, const Vector3& n) const {
        return v - n * 2.0f * v.dot(n);
    }

    bool refract(const Vector3& v, const Vector3& n, float niOverNt, Vector3& refracted) const {
        float dt = v.dot(n);
        float discriminant = 1.0f - niOverNt * niOverNt * (1 - dt * dt);
        if (discriminant > 0) {
            refracted = (v - n * dt) * niOverNt - n * sqrtf(discriminant);
            return true;
        } else {
            return false;
        }
    }

    float schlick(float cosine, float refIdx) const {
        float r0 = (1 - refIdx) / (1 + refIdx);
        r0 *= r0;
        return r0 + (1 - r0) * powf(1 - cosine, 5);
    }
};

// emissive (ls)
class DiffuseLight : public Material {
public:
    std::shared_ptr<Texture> emit;

    DiffuseLight(std::shared_ptr<Texture> a) : emit(a) {}

    virtual bool scatter(const Ray& rayIn, const Vector3& intersection, const Vector3& normal,
                         Vector3& attenuation, Ray& scattered) const {
        return false;
    }

    virtual Vector3 emitted(float u, float v, const Vector3& p) const {
        return emit->value(u, v, p);
    }
};

// ..shape
class Shape {
public:
    virtual bool intersect(const Ray& ray, float tMin, float tMax, float& t, Vector3& normal, std::shared_ptr<Material>& material) const = 0;
    virtual ~Shape() {}
};

// ..sphere
class Sphere : public Shape {
public:
    Vector3 center;
    float radius;
    std::shared_ptr<Material> material;

    Sphere(const Vector3& center, float radius, std::shared_ptr<Material> material)
        : center(center), radius(radius), material(material) {}

    virtual bool intersect(const Ray& ray, float tMin, float tMax, float& t, Vector3& normal, std::shared_ptr<Material>& mat) const {
        Vector3 oc = ray.origin - center;
        float a = ray.direction.dot(ray.direction);
        float bHalf = oc.dot(ray.direction);
        float c = oc.dot(oc) - radius * radius;
        float discriminant = bHalf * bHalf - a * c;

        if (discriminant > 0) {
            float sqrtD = sqrtf(discriminant);
            float root = (-bHalf - sqrtD) / a;
            if (root < tMax && root > tMin) {
                t = root;
                normal = (ray.pointAtParameter(t) - center) / radius;
                mat = material;
                return true;
            }
            root = (-bHalf + sqrtD) / a;
            if (root < tMax && root > tMin) {
                t = root;
                normal = (ray.pointAtParameter(t) - center) / radius;
                mat = material;
                return true;
            }
        }
        return false;
    }
};

// ...plane class
class Plane : public Shape {
public:
    Vector3 point;
    Vector3 normal;
    std::shared_ptr<Material> material;

    Plane(const Vector3& point, const Vector3& normal, std::shared_ptr<Material> material)
        : point(point), normal(normal.normalize()), material(material) {}

    virtual bool intersect(const Ray& ray, float tMin, float tMax, float& t, Vector3& outNormal, std::shared_ptr<Material>& mat) const {
        float denom = normal.dot(ray.direction);
        if (fabs(denom) > EPSILON) {
            t = (point - ray.origin).dot(normal) / denom;
            if (t >= tMin && t <= tMax) {
                outNormal = normal;
                mat = material;
                return true;
            }
        }
        return false;
    }
};

//.......................................| 3 |.......................................
// scene
class Scene {
/*
here we contain a list of objects and lights in the scene.
we also have a function to check for intersections with the objects in the scene.
*/
public:
    std::vector<std::shared_ptr<Shape>> objects;

    bool intersect(const Ray& ray, float tMin, float tMax, float& t, Vector3& normal, std::shared_ptr<Material>& material) const {
        bool hitAnything = false;
        float closestSoFar = tMax;

        for (const auto& object : objects) {
            float tempT;
            Vector3 tempNormal;
            std::shared_ptr<Material> tempMaterial;

            if (object->intersect(ray, tMin, closestSoFar, tempT, tempNormal, tempMaterial)) {
                hitAnything = true;
                closestSoFar = tempT;
                t = tempT;
                normal = tempNormal;
                material = tempMaterial;
            }
        }
        return hitAnything;
    }
};

// camera
class Camera {
/*
the camera class is used to generate rays for rendering the scene.
it has a position, a look-at point, and a field of view.
*/
public:
    Vector3 origin;
    Vector3 lowerLeftCorner;
    Vector3 horizontal;
    Vector3 vertical;
    Vector3 u, v, w;
    float lensRadius;

    Camera(Vector3 lookfrom, Vector3 lookat, Vector3 vup, float vfov, float aspect, float aperture, float focusDist) {
        lensRadius = aperture / 2.0f;
        float theta = degreeToRadians(vfov);
        float halfHeight = tanf(theta / 2);
        float halfWidth = aspect * halfHeight;
        origin = lookfrom;
        w = (lookfrom - lookat).normalize();
        u = vup.cross(w).normalize();
        v = w.cross(u);
        lowerLeftCorner = origin - u * halfWidth * focusDist - v * halfHeight * focusDist - w * focusDist;
        horizontal = u * 2 * halfWidth * focusDist;
        vertical = v * 2 * halfHeight * focusDist;
    }

    Ray getRay(float s, float t) const {
        Vector3 rd = randomInUnitDisk() * lensRadius;
        Vector3 offset = u * rd.x + v * rd.y;
        return Ray(origin + offset, lowerLeftCorner + horizontal * s + vertical * t - origin - offset);
    }

private:
    Vector3 randomInUnitDisk() const {
        Vector3 p;
        do {
            p = Vector3(randomFloat(), randomFloat(), 0) * 2.0f - Vector3(1, 1, 0);
        } while (p.dot(p) >= 1.0f);
        return p;
    }
};

// ...global variables (for rendering)
int g_width = 800;
int g_height = 600;

// mutex for thread synchronization
std::mutex g_mutex;

// global Pixel Data
uint32_t* pixelData = nullptr;

// function Declarations
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
void renderScene(HWND hwnd, HDC hdc, int width, int height);
Vector3 rayColor(const Ray& ray, const Scene& scene, int depth);

// main entry (win)
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    srand(static_cast<unsigned>(time(0)));

    // ..registration of window class
    const TCHAR CLASS_NAME[] = _T("RayTracerWindowClass");
    WNDCLASS wc = {};

    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);

    RegisterClass(&wc);

    // ..creating the window
    HWND hwnd = CreateWindowEx(
        0,
        CLASS_NAME,
        _T("Feritracing | @FECORO"),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, g_width, g_height,
        NULL, NULL, hInstance, NULL
    );

    if (!hwnd) return 0;

    ShowWindow(hwnd, nCmdShow);

    // .message loop
    MSG msg = {};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}

//.......................................| 4 |.......................................
// win procedure
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static HDC hdcMem = NULL;
    static HBITMAP hBitmap = NULL;
    static BITMAPINFO bmi = {};
    static bool rendering = false;

    switch (uMsg) {
    case WM_CREATE: {
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = g_width;
        bmi.bmiHeader.biHeight = g_height; 
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;

        hdcMem = CreateCompatibleDC(NULL);
        hBitmap = CreateDIBSection(hdcMem, &bmi, DIB_RGB_COLORS, (void**)&pixelData, NULL, 0);
        SelectObject(hdcMem, hBitmap);

        // Start rendering thread
        rendering = true;
        std::thread renderThread(renderScene, hwnd, hdcMem, g_width, g_height);
        renderThread.detach();

        break;
    }
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdcWindow = BeginPaint(hwnd, &ps);
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            BitBlt(hdcWindow, 0, 0, g_width, g_height, hdcMem, 0, 0, SRCCOPY);
        }
        EndPaint(hwnd, &ps);
        break;
    }
    case WM_DESTROY: {
        rendering = false;
        DeleteDC(hdcMem);
        DeleteObject(hBitmap);
        PostQuitMessage(0);
        break;
    }
    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}

// ...render scene function
void renderScene(HWND hwnd, HDC hdc, int width, int height) {
    // .initializing scene
    Scene scene;

    // .textures
    auto checkerTexture = std::make_shared<CheckerTexture>(
        std::make_shared<SolidColor>(Vector3(0.8f, 0.8f, 0.8f)),
        std::make_shared<SolidColor>(Vector3(0.1f, 0.1f, 0.1f))
    );

    // ..materials
    auto groundMaterial = std::make_shared<Lambertian>(checkerTexture);
    auto glassMaterial = std::make_shared<Dielectric>(1.5f);
    auto metalMaterial = std::make_shared<Metal>(Vector3(0.8f, 0.6f, 0.2f), 0.0f);
    auto lambertianMaterial = std::make_shared<Lambertian>(Vector3(0.1f, 0.2f, 0.5f));
    auto lightMaterial = std::make_shared<DiffuseLight>(std::make_shared<SolidColor>(Vector3(7, 7, 7)));

    // ..objects
    scene.objects.push_back(std::make_shared<Plane>(Vector3(0, -1, 0), Vector3(0, 1, 0), groundMaterial)); // --reflective floor
    scene.objects.push_back(std::make_shared<Sphere>(Vector3(0.0f, 0.0f, -1.0f), 0.5f, lambertianMaterial));
    scene.objects.push_back(std::make_shared<Sphere>(Vector3(-1.0f, 0.0f, -1.5f), 0.5f, glassMaterial));
    scene.objects.push_back(std::make_shared<Sphere>(Vector3(1.0f, 0.0f, -1.5f), 0.5f, metalMaterial));
    scene.objects.push_back(std::make_shared<Sphere>(Vector3(0.0f, 3.0f, -2.0f), 1.0f, lightMaterial)); // --light source

    // ..camera setup
    Vector3 lookFrom(13, 2, 3);
    Vector3 lookAt(0, 0, 0);
    float distToFocus = 10.0f;
    float aperture = 0.1f;
    Camera camera(lookFrom, lookAt, Vector3(0, 1, 0), 20.0f, float(width) / float(height), aperture, distToFocus);

    // ..image settings
    const int samplesPerPixel = 100; // Adjust for performance
    const int maxDepth = 50;

    // ..rendering setup
    const int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::vector<Vector3> framebuffer(width * height);

    auto renderRowRange = [&](int startY, int endY) {
        for (int j = startY; j < endY; ++j) {
            for (int i = 0; i < width; ++i) {
                Vector3 color(0, 0, 0);
                for (int s = 0; s < samplesPerPixel; ++s) {
                    float u = (i + randomFloat()) / (width - 1);
                    float v = (j + randomFloat()) / (height - 1);
                    Ray ray = camera.getRay(u, v);
                    color += rayColor(ray, scene, maxDepth);
                }
                color /= samplesPerPixel;
                color = Vector3(sqrtf(color.x), sqrtf(color.y), sqrtf(color.z)); // --gamma correction
                framebuffer[j * width + i] = color;
            }

            // ..update display every few rows
            if (j % 10 == 0) {
                std::lock_guard<std::mutex> lock(g_mutex);
                for (int y = startY; y <= j; ++y) {
                    for (int x = 0; x < width; ++x) {
                        Vector3 color = framebuffer[y * width + x];
                        uint8_t r = static_cast<uint8_t>(clamp(color.x, 0.0f, 1.0f) * 255.99f);
                        uint8_t g = static_cast<uint8_t>(clamp(color.y, 0.0f, 1.0f) * 255.99f);
                        uint8_t b = static_cast<uint8_t>(clamp(color.z, 0.0f, 1.0f) * 255.99f);
                        pixelData[y * width + x] = (r << 16) | (g << 8) | b;
                    }
                }
                InvalidateRect(hwnd, NULL, FALSE);
            }
        }
    };

    // ...launch the rendering threads
    int rowsPerThread = height / numThreads;
    for (int t = 0; t < numThreads; ++t) {
        int startRow = t * rowsPerThread;
        int endRow = (t == numThreads - 1) ? height : startRow + rowsPerThread;
        threads.emplace_back(renderRowRange, startRow, endRow);
    }

    // .. now join threads
    for (auto& thread : threads) {
        thread.join();
    }

    // ...the final update
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Vector3 color = framebuffer[y * width + x];
                uint8_t r = static_cast<uint8_t>(clamp(color.x, 0.0f, 1.0f) * 255.99f);
                uint8_t g = static_cast<uint8_t>(clamp(color.y, 0.0f, 1.0f) * 255.99f);
                uint8_t b = static_cast<uint8_t>(clamp(color.z, 0.0f, 1.0f) * 255.99f);
                pixelData[y * width + x] = (r << 16) | (g << 8) | b;
            }
        }
        InvalidateRect(hwnd, NULL, FALSE);
    }
}

// ...compute color for a ray
Vector3 rayColor(const Ray& ray, const Scene& scene, int depth) {
    if (depth <= 0)
        return Vector3(0, 0, 0);

    float t;
    Vector3 normal;
    std::shared_ptr<Material> material;
    if (scene.intersect(ray, EPSILON, INFINITY, t, normal, material)) {
        Ray scattered;
        Vector3 attenuation;
        Vector3 emitted = material->emitted(0, 0, ray.pointAtParameter(t));
        if (material->scatter(ray, ray.pointAtParameter(t), normal, attenuation, scattered))
            return emitted + attenuation * rayColor(scattered, scene, depth - 1);
        else
            return emitted;
    } else {
        // ..sky background (lerp)
        Vector3 unitDirection = ray.direction.normalize();
        float t = 0.5f * (unitDirection.y + 1.0f);
        return Vector3(0.7f, 0.8f, 1.0f) * (1.0f - t) + Vector3(1.0f, 1.0f, 1.0f) * t;
    }
}
//.......................................| 5 |.......................................
//@FECORO, all rights reserved (2022)
