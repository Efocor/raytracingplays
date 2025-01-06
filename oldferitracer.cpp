//.......................................| 1 |.......................................
/*
* -- Second of 3 ray tracing implementations --
*
 * Raytracer Implementation / Timed Visualization @FECORO
 *
 * This code implements a real-time ray tracer using a physically-based rendering approach.
 * It supports various materials like Lambertian (diffuse), Metal (reflective), and Dielectric (transparent).
 * The ray tracer is multi-threaded to leverage modern CPU cores for faster rendering.
 *
 * Key Features:
 * - Ray-sphere intersection for rendering spheres.
 * - Support for Lambertian, Metal, and Dielectric materials.
 * - Depth-limited recursion for ray tracing.
 * - Anti-aliasing through multi-sampling.
 * - Gamma correction for accurate color representation.
 * - Real-time display using a Windows GUI.
 *
 * The code is structured into several components:
 * - Vector3: A 3D vector class for mathematical operations.
 * - Ray: Represents a ray with an origin and direction.
 * - Shape: Base class for geometric shapes (e.g., Sphere).
 * - Material: Base class for materials (e.g., Lambertian, Metal, Dielectric).
 * - Scene: Manages objects and lights in the scene.
 * - Camera: Defines the viewpoint and generates rays.
 * - Windows GUI: Handles rendering and display using GDI.
 *
 * Usage:
 * - The program creates a window and renders the scene in real-time.
 * - The scene includes a ground plane and three spheres with different materials.
 * - The camera is positioned to view the scene from a specific angle.
 * - The rendering is multi-threaded for performance.
 *
 * It is based on other already known ways of ray tracing.
 * By Felipe Alexander Correa Rodríguez.
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

//...constants
const float PI = 3.14159265358979323846f;
const int MAX_DEPTH = 50;

//...utility Functions
float clamp(float value, float minValue, float maxValue) {
    return value < minValue ? minValue : (value > maxValue ? maxValue : value);
}

float randomFloat() {
    return static_cast<float>(rand()) / RAND_MAX;
}

//...vector3 Struct
struct Vector3 {
    float x, y, z;

    Vector3(float x = 0, float y = 0, float z = 0)
        : x(x), y(y), z(z) {}

    //...operator overloads
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

    //...vector Operations
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

//...ray Struct
struct Ray {
    Vector3 origin, direction;

    Ray(const Vector3& origin, const Vector3& direction)
        : origin(origin), direction(direction.normalize()) {}

    Vector3 pointAtParameter(float t) const {
        return origin + direction * t;
    }
};

//...forward declarations
class Scene;
class Material;
class Light;

//...shape
class Shape {
public:
    virtual float intersect(const Ray& ray, Vector3& normal) const = 0;
};

//...material Base Class
class Material {
public:
    virtual Vector3 shade(const Ray& ray, const Vector3& intersection, const Vector3& normal,
                          const std::vector<std::shared_ptr<Light>>& lights, float ambientLight,
                          const Scene& scene, int depth) const = 0;
};

//...light Class
class Light {
public:
    Vector3 position;
    Vector3 color;
    float intensity;

    Light(const Vector3& position, const Vector3& color, float intensity)
        : position(position), color(color), intensity(intensity) {}
};

//...scene Class
class Scene {
public:
    std::vector<std::pair<std::shared_ptr<Shape>, std::shared_ptr<Material>>> objects;
    std::vector<std::shared_ptr<Light>> lights;

    Vector3 traceRay(const Ray& ray, int depth) const {
        if (depth >= MAX_DEPTH) return Vector3(0, 0, 0);

        float t = INFINITY;
        Vector3 normal;
        const Material* material = nullptr;

        for (const auto& object : objects) {
            Vector3 tempNormal;
            float tempT = object.first->intersect(ray, tempNormal);
            if (tempT > 0.001f && tempT < t) {
                t = tempT;
                normal = tempNormal;
                material = object.second.get();
            }
        }

        if (material) {
            Vector3 intersection = ray.pointAtParameter(t);
            return material->shade(ray, intersection, normal, lights, 0.1f, *this, depth);
        }

        //...background color gradient
        Vector3 unitDirection = ray.direction.normalize();
        float t_bg = 0.5f * (unitDirection.y + 1.0f);
        return Vector3(1.0f, 1.0f, 1.0f) * (1.0f - t_bg) + Vector3(0.5f, 0.7f, 1.0f) * t_bg;
    }
};

//...helper func
Vector3 reflect(const Vector3& v, const Vector3& n) {
    return v - n * 2.0f * v.dot(n);
}

bool refract(const Vector3& v, const Vector3& n, float niOverNt, Vector3& refracted) {
    Vector3 uv = v.normalize();
    float dt = uv.dot(n);
    float discriminant = 1.0f - niOverNt * niOverNt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = (uv - n * dt) * niOverNt - n * sqrtf(discriminant);
        return true;
    } else {
        return false;
    }
}

float schlick(float cosine, float refIdx) {
    float r0 = (1 - refIdx) / (1 + refIdx);
    r0 *= r0;
    return r0 + (1 - r0) * powf(1 - cosine, 5);
}

Vector3 randomInUnitSphere() {
    Vector3 p;
    do {
        p = Vector3(randomFloat(), randomFloat(), randomFloat()) * 2.0f - Vector3(1, 1, 1);
    } while (p.length() >= 1.0f);
    return p;
}

Vector3 randomInUnitDisk() {
    Vector3 p;
    do {
        p = Vector3(randomFloat(), randomFloat(), 0) * 2.0f - Vector3(1, 1, 0);
    } while (p.dot(p) >= 1.0f);
    return p;
}

//.......................................| 3 |.......................................
//...material Implementations

//...lambertian Material
class Lambertian : public Material {
public:
    Vector3 albedo;

    Lambertian(const Vector3& albedo)
        : albedo(albedo) {}

    Vector3 shade(const Ray& ray, const Vector3& intersection, const Vector3& normal,
                  const std::vector<std::shared_ptr<Light>>& lights, float ambientLight,
                  const Scene& scene, int depth) const override {
        Vector3 target = intersection + normal + randomInUnitSphere();
        Ray scattered(intersection + normal * 1e-4f, target - intersection);
        return albedo * scene.traceRay(scattered, depth + 1);
    }
};

//...metal material
class Metal : public Material {
public:
    Vector3 albedo;
    float fuzz;

    Metal(const Vector3& albedo, float fuzz)
        : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    Vector3 shade(const Ray& ray, const Vector3& intersection, const Vector3& normal,
                  const std::vector<std::shared_ptr<Light>>& lights, float ambientLight,
                  const Scene& scene, int depth) const override {
        Vector3 reflected = reflect(ray.direction.normalize(), normal);
        reflected += randomInUnitSphere() * fuzz;
        Ray scattered(intersection + normal * 1e-4f, reflected);

        if (scattered.direction.dot(normal) > 0)
            return albedo * scene.traceRay(scattered, depth + 1);
        else
            return Vector3(0, 0, 0);
    }
};

//...dielectric (glass)
class Dielectric : public Material {
public:
    float refIdx;

    Dielectric(float refIdx)
        : refIdx(refIdx) {}

    Vector3 shade(const Ray& ray, const Vector3& intersection, const Vector3& normal,
                  const std::vector<std::shared_ptr<Light>>& lights, float ambientLight,
                  const Scene& scene, int depth) const override {
        Vector3 outwardNormal;
        Vector3 reflected = reflect(ray.direction, normal);
        float niOverNt;
        float cosine;
        Vector3 refracted;
        float reflectProb;

        if (ray.direction.dot(normal) > 0) {
            outwardNormal = -normal;
            niOverNt = refIdx;
            cosine = refIdx * ray.direction.dot(normal) / ray.direction.length();
        } else {
            outwardNormal = normal;
            niOverNt = 1.0f / refIdx;
            cosine = -ray.direction.dot(normal) / ray.direction.length();
        }

        if (refract(ray.direction, outwardNormal, niOverNt, refracted)) {
            reflectProb = schlick(cosine, refIdx);
        } else {
            reflectProb = 1.0f;
        }

        if (randomFloat() < reflectProb) {
            Ray reflectedRay(intersection + normal * 1e-4f, reflected);
            return scene.traceRay(reflectedRay, depth + 1);
        } else {
            Ray refractedRay(intersection - outwardNormal * 1e-4f, refracted);
            return scene.traceRay(refractedRay, depth + 1);
        }
    }
};

//...sphere Class
class Sphere : public Shape {
public:
    Vector3 center;
    float radius;

    Sphere(const Vector3& center, float radius)
        : center(center), radius(radius) {}

    float intersect(const Ray& ray, Vector3& normal) const override {
        //...ray-sphere intersection algorithm
        Vector3 oc = ray.origin - center;
        float a = ray.direction.dot(ray.direction);
        float b = oc.dot(ray.direction);
        float c = oc.dot(oc) - radius * radius;
        float discriminant = b * b - a * c;

        if (discriminant < 0) return -1.0f;
        float sqrtDisc = sqrtf(discriminant);
        float t = (-b - sqrtDisc) / a;
        if (t < 0.001f) t = (-b + sqrtDisc) / a;
        if (t < 0.001f) return -1.0f;

        normal = (ray.pointAtParameter(t) - center).normalize();
        return t;
    }
};

//...camera Class
class Camera {
public:
    Vector3 origin;
    Vector3 lowerLeftCorner;
    Vector3 horizontal;
    Vector3 vertical;
    Vector3 u, v, w;
    float lensRadius;

    Camera(Vector3 lookfrom, Vector3 lookat, Vector3 vup, float vfov, float aspect, float aperture, float focusDist) {
        lensRadius = aperture / 2.0f;
        float theta = vfov * PI / 180.0f;
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

//...global Variables
int g_width = 800;
int g_height = 600;

//...mutex for thread synchronization
std::mutex g_mutex;

//...global Pixel Data
uint32_t* pixelData = nullptr; //...Define pixelData at global scope

//...forward decla
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
void renderScene(HWND hwnd, HDC hdc, int width, int height);

//...main entry
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    srand(static_cast<unsigned>(time(0)));

    //...registering win class
    const TCHAR CLASS_NAME[] = _T("RayTracerWindowClass");
    WNDCLASS wc = {};

    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);

    RegisterClass(&wc);

    //...creating the Window
    HWND hwnd = CreateWindowEx(
        0,
        CLASS_NAME,
        _T("OLD Feritracer | @FECORO"),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, g_width, g_height,
        NULL, NULL, hInstance, NULL
    );

    if (!hwnd) return 0;

    ShowWindow(hwnd, nCmdShow);

    //...message loop
    MSG msg = {};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}

//...window proc
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static HDC hdcMem = NULL;
    static HBITMAP hBitmap = NULL;
    static BITMAPINFO bmi = {};
    static bool rendering = false;

    switch (uMsg) {
    case WM_CREATE: {
        //..initializing drawing resources
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = g_width;
        bmi.bmiHeader.biHeight = g_height; 
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;

        hdcMem = CreateCompatibleDC(NULL);
        hBitmap = CreateDIBSection(hdcMem, &bmi, DIB_RGB_COLORS, (void**)&pixelData, NULL, 0);
        SelectObject(hdcMem, hBitmap);

        //...start rendering thread
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

//.......................................| 4 |.......................................
//...render scene func
void renderScene(HWND hwnd, HDC hdc, int width, int height) {
    //...Initialize scene
    Scene scene;

    //...add objects
    auto groundMaterial = std::make_shared<Lambertian>(Vector3(0.8f, 0.8f, 0.0f));
    auto centerMaterial = std::make_shared<Lambertian>(Vector3(0.1f, 0.2f, 0.5f));
    auto leftMaterial = std::make_shared<Dielectric>(1.5f);
    auto rightMaterial = std::make_shared<Metal>(Vector3(0.8f, 0.6f, 0.2f), 0.0f);

    scene.objects.push_back({ std::make_shared<Sphere>(Vector3(0.0f, -1000.0f, -1.0f), 1000.0f), groundMaterial });
    scene.objects.push_back({ std::make_shared<Sphere>(Vector3(0.0f, 0.0f, -1.0f), 0.5f), centerMaterial });
    scene.objects.push_back({ std::make_shared<Sphere>(Vector3(-1.0f, 0.0f, -1.0f), 0.5f), leftMaterial });
    scene.objects.push_back({ std::make_shared<Sphere>(Vector3(1.0f, 0.0f, -1.0f), 0.5f), rightMaterial });

    //...initialize camera
    Vector3 lookFrom(3, 3, 2);
    Vector3 lookAt(0, 0, -1);
    float distToFocus = (lookFrom - lookAt).length();
    float aperture = 0.1f;
    Camera camera(lookFrom, lookAt, Vector3(0, 1, 0), 20, float(width) / float(height), aperture, distToFocus);

    //...image settings
    const int samplesPerPixel = 100;
    const int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    //...framebuffer
    std::vector<Vector3> framebuffer(width * height);

    //...render function for threads
    auto renderRowRange = [&](int startRow, int endRow) {
        for (int j = startRow; j < endRow; ++j) {
            for (int i = 0; i < width; ++i) {
                Vector3 color(0, 0, 0);
                for (int s = 0; s < samplesPerPixel; ++s) {
                    float u = float(i + randomFloat()) / float(width);
                    float v = float(j + randomFloat()) / float(height);
                    Ray ray = camera.getRay(u, v);
                    color += scene.traceRay(ray, 0);
                }
                color /= float(samplesPerPixel);
                //...gamma correction
                color = Vector3(sqrtf(color.x), sqrtf(color.y), sqrtf(color.z));
                framebuffer[j * width + i] = color;
            }

            //...updating the display every few rows
            if (j % 10 == 0) {
                std::lock_guard<std::mutex> lock(g_mutex);
                for (int y = startRow; y <= j; ++y) {
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

    //...start threads
    int rowsPerThread = height / numThreads;
    for (int t = 0; t < numThreads; ++t) {
        int startRow = t * rowsPerThread;
        int endRow = (t == numThreads - 1) ? height : startRow + rowsPerThread;
        threads.emplace_back(renderRowRange, startRow, endRow);
    }

    //...join such threads
    for (auto& thread : threads) {
        thread.join();
    }

    //...final update
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
//.......................................| 5 |.......................................
//@FECORO, all rights reserved.
