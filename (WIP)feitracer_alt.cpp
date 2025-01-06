//............................................| CODE Start |............................................
//.. --- 3rd implementation of raytracing capabilities --- .. (this is a wip)

/*
 * Path Tracer Implementation with Advanced Features
 * 
 * This code implements a path tracer, a type of ray tracer that simulates global illumination
 * by tracing rays of light as they bounce around a scene. It includes advanced features such
 * as support for different material types (diffuse, specular, refractive), texture mapping
 * (solid, checker, noise, and image-based), and parallel rendering using OpenMP.
 * 
 * Key Features:
 * - Physically-based rendering using Monte Carlo path tracing.
 * - Support for diffuse, specular, and refractive materials.
 * - Texture mapping with solid colors, checker patterns, procedural noise, and image textures.
 * - Parallel rendering using OpenMP for improved performance.
 * - Depth-limited recursion with Russian Roulette termination for efficiency.
 * - Support for loading image textures in PPM format.
 * - Realistic lighting with emissive materials and area lights.
 * 
 * The code is structured into several components:
 * - Vec: A 3D vector class for mathematical operations.
 * - Ray: Represents a ray with an origin and direction.
 * - Material: Defines material properties, including reflection type, color, emission, and texture.
 * - Sphere: Represents a sphere in the scene with a position, radius, and material.
 * - Scene: Manages the collection of spheres and provides functions for ray intersection and rendering.
 * - RNG: A random number generator for Monte Carlo sampling.
 * 
 * Usage:
 * - The program renders a scene with multiple spheres, each with different materials and textures.
 * - The output is saved as a PPM image file.
 * - The number of samples per pixel can be specified as a command-line argument.
 * 
 * Differences from the previous implementations of Feritracer:
 * - This implementation includes texture mapping (checker, noise, and image-based textures).
 * - It uses OpenMP for parallel rendering, significantly improving performance.
 * - It supports more complex materials, including refractive (glass) and emissive (light) materials.
 * - The scene setup is more detailed, with additional spheres and textures for a richer visual result.
 * - The code is modular and extensible, making it easier to add new features or modify existing ones.

 By Feri.
 */

//............................................| 2 |............................................

//............. head stack
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <ctime>
#include <omp.h>
#include <filesystem> //... requires c++17

//............. constants and others
namespace PathTracer {

    //...constants
    const double PI = 3.14159265358979323846;
    const double EPSILON = 1e-4;
    const int MAX_DEPTH = 10;

    //...func utilitaries

    //...limita un valor entre 0 y 1
    inline double clamp(double x) {
        return x < 0 ? 0 : (x > 1 ? 1 : x);
    }

    //...convierte un componente de color en double a entero [0,255]
    inline int toInt(double x) { 
        return int(pow(clamp(x), 1 / 2.2) * 255 + 0.5); 
    }

    //...generador de Números Aleatorios Congruente Lineal (LCG)
    struct RNG {
        unsigned long long seed;

        //...constructor que inicializa la semilla
        RNG(unsigned long long s = 0) : seed(s) {}

        //...generamos un número aleatorio en [0,1)
        double rand_double() {
            seed = (seed * 6364136223846793005ULL + 1);
            return (double)(seed >> 33) * (1.0 / (1ULL << 31));
        }
    };

    //...estructura de vector para cálculos 3D
    struct Vec { 
        double x, y, z; 

        //...constructores
        Vec(double x_=0, double y_=0, double z_=0) : x(x_), y(y_), z(z_) {}
        Vec(const Vec &v) : x(v.x), y(v.y), z(v.z) {} //...Constructor de copia

        //...sobrecarga de operadores para aritmética de vectores
        Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
        Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
        Vec operator*(double b) const { return Vec(x * b, y * b, z * b); }
        Vec mult(const Vec &b) const { return Vec(x * b.x, y * b.y, z * b.z); }

        //...normaliza el vector
        Vec& norm() { 
            double length = sqrt(x * x + y * y + z * z); 
            return *this = *this * (1 / length); 
        }

        //...producto punto
        double dot(const Vec &b) const { return x * b.x + y * b.y + z * b.z; }

        //...producto cruz
        Vec operator%(const Vec& b) const { 
            return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); 
        }
    };

    //...estructura de Rayo
    struct Ray { 
        Vec o, d; 
        Ray(const Vec &o_, const Vec &d_) : o(o_), d(d_) {} 
    };

    //...tipos de reflexión
    enum Refl_t { DIFF, SPEC, REFR, TEXTURE }; //...Añadido TEXTURE para mapeo de texturas

    //...tipos de textura
    enum TexType { SOLID, CHECKER, NOISE, IMAGE };

    //...estructura de textura
    struct Texture {
        TexType type;          //... * Tipo de textura
        Vec color1, color2;    //... * Colores utilizados en las texturas
        double scale;          //... * Factor de escala para texturas
        //...para texturas de imagen
        int img_width, img_height;
        std::vector<Vec> img_data;

        //...constructor que inicializa valores predeterminados
        Texture() : type(SOLID), color1(Vec()), color2(Vec()), scale(1.0), img_width(0), img_height(0) {}
    };

    //...estructura de material
    struct Material {
        Refl_t refl;      //... * Tipo de reflexión
        Vec color;        //... * Color del material
        Vec emission;     //... * Color de emisión (para fuentes de luz)
        Texture texture;  //... * Propiedades de textura

        //... constructores
        Material() : refl(DIFF), color(Vec()), emission(Vec()), texture() {}
        Material(Refl_t refl_, const Vec &color_, const Vec &emission_) 
            : refl(refl_), color(color_), emission(emission_), texture() {}

        //...método para obtener color basado en el tipo de textura
        Vec getColor(const Vec &hit_point, const Vec &normal) const {
            switch (texture.type) {
                case SOLID:
                    return color;
                case CHECKER: {
                    double s = texture.scale;
                    int chk = (int(floor(hit_point.x * s)) + 
                               int(floor(hit_point.y * s)) + 
                               int(floor(hit_point.z * s))) & 1;
                    return chk ? texture.color1 : texture.color2;
                }
                case NOISE: {
                    //... una textura simple basada en ruido usando funciones seno
                    double noise = sin(hit_point.x * texture.scale) * 
                                   sin(hit_point.y * texture.scale) * 
                                   sin(hit_point.z * texture.scale);
                    noise = (noise + 1) / 2; //...Normaliza a [0,1]
                    return color * noise + texture.color1 * (1 - noise);
                }
                case IMAGE:
                    if (texture.img_width > 0 && texture.img_height > 0) {
                        //...mapeo esférico simple para texturas de imagen
                        double u = 0.5 + atan2(normal.z, normal.x) / (2 * PI);
                        double v = 0.5 - asin(normal.y) / PI;
                        int img_x = std::min(std::max(int(u * texture.img_width), 0), texture.img_width - 1);
                        int img_y = std::min(std::max(int(v * texture.img_height), 0), texture.img_height - 1);
                        return texture.img_data[img_y * texture.img_width + img_x];
                    }
                    //...fallback si la imagen no está cargada correctamente
                    return color;
                default:
                    return color;
            }
        }
    };

    //...estructura para esfera
    struct Sphere {
        double rad;    //... * Radio
        Vec p;         //... * Posición
        Vec e;         //... * Emisión
        Material mat;  //... * Material

        //...constructor
        Sphere(double rad_, const Vec &p_, const Vec &e_, const Material &mat_) 
            : rad(rad_), p(p_), e(e_), mat(mat_) {}

        //...intersección ray-sphere
        double intersect(const Ray &r) const { 
            Vec op = p - r.o; 
            double t, eps = EPSILON, b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;
            if (det < 0) return 0; else det = sqrt(det);
            return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
        }
    };

    //...lista global de esferas que componen la escena
    std::vector<Sphere> spheres;

    //...función pa cargar texturas de imagen (formato PPM P3)
    bool loadImageTexture(const std::string &filename, Texture &texture) {
        //...comprobar si el archivo existe
        if (!std::filesystem::exists(filename)) {
            std::cerr << "Archivo de textura no encontrado: " << filename << std::endl;
            return false;
        }

        std::ifstream infile(filename.c_str());
        if (!infile.is_open()) {
            std::cerr << "No se puede abrir el archivo de textura: " << filename << std::endl;
            return false;
        }

        std::string header;
        infile >> header;
        if (header != "P3") {
            std::cerr << "Formato de imagen no soportado (solo se admite P3 PPM): " << filename << std::endl;
            return false;
        }

        infile >> texture.img_width >> texture.img_height;
        int max_val;
        infile >> max_val;

        texture.img_data.reserve(texture.img_width * texture.img_height);
        int r, g, b;
        while (infile >> r >> g >> b) {
            texture.img_data.emplace_back(Vec(r / 255.0, g / 255.0, b / 255.0));
        }

        infile.close();
        std::cout << "Textura cargada exitosamente: " << filename << std::endl;
        return true;
    }

    //...función para inicializar la escena con detalles mejorados
    void initScene() {
        
        //...Limpiar cualquier esfera existente
        spheres.clear();
        
        //...===============================
        //...** Suelo y Ambiente Básico **
        //...===============================
        
        //...suelo que simula una vasta pradera con textura CHECKER
        Material ground_material;
        ground_material.refl = DIFF;
        ground_material.color = Vec(0.5, 0.5, 0.5);
        ground_material.texture.type = CHECKER;
        ground_material.texture.color1 = Vec(0.9, 0.9, 0.9);
        ground_material.texture.color2 = Vec(0.1, 0.1, 0.1);
        ground_material.texture.scale = 20.0;
        spheres.emplace_back(1e5, Vec(50, -1e5, 81.6), Vec(), ground_material); //...plano de Suelo Místico
        
        //...===============================
        //...** Estructuras de Paredes Místicas **
        //...===============================
        
        //...utilizamos esferas más pequeñas para formar paredes con patrones
        //...pared izquierda - patrón vertical de esferas
        for(int i = 0; i < 10; ++i) {
            Material wall_material;
            wall_material.refl = DIFF;
            wall_material.color = Vec(0.2, 0.3, 0.5 + 0.05 * i); //...cambio de color para crear un degradado
            spheres.emplace_back(5.0, Vec(5, 10 + i * 8, 81.6), Vec(), wall_material); //...esferas pared izquierda
        }
        
        //...pared derecha - patrón horizontal de esferas con Textura NOISE
        for(int i = 0; i < 15; ++i) {
            Material noise_wall_material;
            noise_wall_material.refl = DIFF;
            noise_wall_material.color = Vec(0.3, 0.6, 0.4);
            noise_wall_material.texture.type = NOISE;
            noise_wall_material.texture.color1 = Vec(0.3, 0.6, 0.4);
            noise_wall_material.texture.color2 = Vec(1, 1, 1);
            noise_wall_material.texture.scale = 5.0;
            spheres.emplace_back(4.0, Vec(95, 8, 80 + i * 5), Vec(), noise_wall_material); //...Esferas Pared Derecha con Textura
        }
        
        //...pared Trasera - arcos formados por Esferas
        for(int i = 0; i < 20; ++i) {
            double angle = (PI / 10) * i; //...distribución en arco de 180 grados (sino se sale)
            double radius = 30.0;
            Vec position(50 + radius * cos(angle), 40 + radius * sin(angle), 160.0);
            Material arch_material;
            arch_material.refl = DIFF;
            arch_material.color = Vec(0.5, 0.4, 0.6);
            spheres.emplace_back(3.0, position, Vec(), arch_material); //...esferas de arco trasero
        }
        
        //...pared frontal - patrón radial de esferas
        for(int i = 0; i < 12; ++i) {
            double angle = (2 * PI / 12) * i;
            double radius = 20.0;
            Vec position(50 + radius * cos(angle), 20, -30.0);
            Material radial_material;
            radial_material.refl = DIFF;
            radial_material.color = Vec(0.7, 0.2, 0.5);
            spheres.emplace_back(4.0, position, Vec(), radial_material); //...esferas patrón radial frontal
        }
        
        //...===============================
        //...** Estructuras Decorativas Místicas **
        //...===============================
        
        //...esfera metálica
        Material metallic_material;
        metallic_material.refl = SPEC;
        metallic_material.color = Vec(0.6, 0.4, 0.8); //...color púrpura metálico
        metallic_material.texture.type = SOLID;
        spheres.emplace_back(16.5, Vec(30, 20, 60), Vec(), metallic_material); //...esferita Metálica Mística
        
        //...esfera de textura checker
        Material checker_material;
        checker_material.refl = DIFF;
        checker_material.color = Vec(1, 1, 1);
        checker_material.texture.type = CHECKER;
        checker_material.texture.color1 = Vec(0.3, 0.7, 0.5); //...verde místico
        checker_material.texture.color2 = Vec(0.1, 0.2, 0.1); //...verde oscuro
        checker_material.texture.scale = 10.0;
        spheres.emplace_back(16.5, Vec(40, 20, 30), Vec(), checker_material); //...esfera checker
        
        //...esfera de textura noise
        Material noise_material;
        noise_material.refl = DIFF;
        noise_material.color = Vec(0.4, 0.7, 0.2);
        noise_material.texture.type = NOISE;
        noise_material.texture.color1 = Vec(0.4, 0.7, 0.2);
        noise_material.texture.color2 = Vec(1, 1, 1);
        noise_material.texture.scale = 5.0;
        spheres.emplace_back(16.5, Vec(60, 20, 50), Vec(), noise_material); //...Esfera Noise Mística
        
        //...esfera metálica adicional para detalles
        Material metallic_detail;
        metallic_detail.refl = SPEC;
        metallic_detail.color = Vec(0.8, 0.6, 0.8); //...color metálico rosado
        metallic_detail.texture.type = SOLID;
        spheres.emplace_back(12.0, Vec(70, 25, 70), Vec(), metallic_detail); //...esfera Metálica Detalle
        
        //...esfera dieléctrica con Textura NOISE (removida para mantener simplicidad)
        
        //...===============================
        //...     ** Fuentes de Luz **
        //...===============================
        
        //...fuente de luz principal: ethereal
        Material light_material;
        light_material.refl = DIFF;
        light_material.color = Vec();
        light_material.emission = Vec(15, 15, 15); //...luz blanca intensa y suave
        spheres.emplace_back(600, Vec(50, 681.6 - 0.27, 81.6), Vec(), light_material); //...fuente de luz principal
        
        //...esferas de luz pequeñas (orbes flotantes)
        Material small_light_material;
        small_light_material.refl = DIFF;
        small_light_material.color = Vec();
        small_light_material.emission = Vec(4, 4, 4); //...luz más pequeña y suave
        spheres.emplace_back(10.0, Vec(50, 25, 100), Vec(), small_light_material); //...esfera de luz pequeña
        
        //...===============================
        //...** Puntos de Luz Decorativos **
        //...===============================
        
        //...esfera decorativa con Textura Checker
        Material decorative_material;
        decorative_material.refl = DIFF;
        decorative_material.color = Vec(0.3, 0.7, 0.5);
        decorative_material.texture.type = CHECKER;
        decorative_material.texture.color1 = Vec(0.3, 0.7, 0.5);
        decorative_material.texture.color2 = Vec(0.1, 0.2, 0.1);
        decorative_material.texture.scale = 15.0;
        spheres.emplace_back(16.5, Vec(100, 20, 100), Vec(), decorative_material); //...esfera decorativa Checker
        
    }

    //...función para verificar intersecciones de rayos con esferas
    inline bool intersect(const Ray &r, double &t, int &id) {
        double inf = t = 1e20;
        for(int i = 0; i < spheres.size(); i++) {
            double d = spheres[i].intersect(r);
            if(d != 0 && d < t) {
                t = d;
                id = i;
            }
        }
        return t < inf;
    }

    //...función radiance: ->calcula el color visto por un rayo
    Vec radiance(const Ray &r, int depth, RNG &rng) {
        double t; //...distancia a la intersección
        int id = 0; //...id del objeto intersectado

        if (!intersect(r, t, id)) return Vec(); //...si no hay intersección, devuelve negro

        const Sphere &obj = spheres[id]; //...el objeto golpeado
        Vec x = r.o + r.d * t; //...punto de intersección
        Vec n = (x - obj.p).norm(); //...normal en el punto de intersección
        Vec nl = n.dot(r.d) < 0 ? n : n * -1; //...normal orientada

        Vec f = obj.mat.getColor(x, n); //...obtener color del material basado en la textura

        double p = std::max({f.x, f.y, f.z}); //...máximo reflejo

        //...terminación por Ruleta Rusa
        if (++depth > MAX_DEPTH) {
            if (rng.rand_double() < p)
                f = f * (1.0 / p);
            else
                return obj.mat.emission;
        }

        //...manejo del material basado en el rayo
        if (obj.mat.refl == DIFF) { //...teflexión DIFFUSA IDEAL
            double r1 = 2 * PI * rng.rand_double();
            double r2 = rng.rand_double();
            double r2s = sqrt(r2);

            //...base ortonormal
            Vec w = nl;
            Vec u = ((fabs(w.x) > 0.1 ? Vec(0, 1, 0) : Vec(1, 0, 0)) % w).norm();
            Vec v = w % u;

            Vec d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();
            return obj.mat.emission + f.mult(radiance(Ray(x, d), depth, rng));
        } 
        else if (obj.mat.refl == SPEC) { //...Reflexión ESPECULAR IDEAL
            Vec reflected_dir = r.d - n * 2 * n.dot(r.d);
            return obj.mat.emission + f.mult(radiance(Ray(x, reflected_dir), depth, rng));
        } 
        else if (obj.mat.refl == REFR) { //...REFRACTACIÓN DIELECTRICA IDEAL
            Ray reflRay(x, r.d - n * 2 * n.dot(r.d)); //...Rayo de reflexión
            bool into = n.dot(nl) > 0; //...¿Rayo desde afuera hacia adentro?
            double nc = 1; //... * refractivo del aire
            double nt = 1.5; //...Índice refractivo del vidrio
            double nnt = into ? nc / nt : nt / nc;
            double ddn = r.d.dot(nl);
            double cos2t;
            if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0) //...Reflexión interna total
                return obj.mat.emission + f.mult(radiance(reflRay, depth, rng));
            Vec tdir = (r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm(); //...Dirección de transmisión

            double a = nt - nc, b = nt + nc;
            double R0 = (a * a) / (b * b);
            double c = 1 - (into ? -ddn : tdir.dot(n));
            double Re = R0 + (1 - R0) * pow(c, 5); //...reflectancia de Fresnel
            double Tr = 1 - Re;
            double P = 0.25 + 0.5 * Re;
            double RP = Re / P;
            double TP = Tr / (1 - P);

            //...ruleta Rusa
            if (depth > 2) {
                if (rng.rand_double() < P)
                    return obj.mat.emission + f.mult(radiance(reflRay, depth, rng) * RP);
                else
                    return obj.mat.emission + f.mult(radiance(Ray(x, tdir), depth, rng) * TP);
            }
            //...de lo contrario, retornar la reflexión y refracción de Fresnel
            return obj.mat.emission + f.mult(radiance(reflRay, depth, rng) * Re + radiance(Ray(x, tdir), depth, rng) * Tr);
        }

        //...caso por defecto (no debería alcanzar aquí)
        return Vec();
    }

    //...función principal de renderizado
    void render(int argc, char *argv[]) {
        //...inicializar la escena
        initScene();

        //...parámetros de la imagen
        int w = 1024, h = 768;
        int samps = (argc >= 2) ? atoi(argv[1]) : 1000; //...número de muestras por píxel (default 1000)

        //...configuración de la cámara
        Vec cam_origin(50, 52, 295.6);
        Vec cam_dir = Vec(0, -0.042612, -1).norm();
        Vec cx = Vec(w * 0.5135 / h, 0, 0);
        Vec cy = (cx % cam_dir).norm() * 0.5135;

        //...buffer de la imagen
        std::vector<Vec> c(w * h, Vec());

        //...semilla para RNG basada en el tiempo actual
        unsigned long long seed = static_cast<unsigned long long>(time(0));

        //...paralelización con OpenMP
        #pragma omp parallel for schedule(dynamic, 1) collapse(2)
        for(int y = 0; y < h; y++) { //...bucle sobre filas de la imagen
            for(int x = 0; x < w; x++) { //...bucle sobre columnas de la imagen
                int i = (h - y - 1) * w + x; //...indice del píxel
                Vec pixel_color;

                //...inicializar RNG con una semilla única para cada píxel
                RNG rng(seed + x + y * w);

                for(int sy = 0; sy < 2; sy++) { //...2x2 subpíxeles
                    for(int sx = 0; sx < 2; sx++) { //...2x2 subpíxeles
                        for(int s = 0; s < samps; s++) { //...muestras por subpíxel
                            double r1 = 2 * rng.rand_double();
                            double dx = (r1 < 1) ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                            double r2 = 2 * rng.rand_double();
                            double dy = (r2 < 1) ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                            Vec d = cx * (((sx + 0.5 + dx) / 2 + x) / w - 0.5) +
                                     cy * (((sy + 0.5 + dy) / 2 + y) / h - 0.5) + cam_dir;
                            d = d.norm();
                            pixel_color = pixel_color + radiance(Ray(cam_origin + d * 140.0, d), 0, rng) * (1.0 / samps);
                        }
                    }
                }
                //...promediar y agregar colores al buffer
                c[i] = c[i] + Vec(clamp(pixel_color.x), clamp(pixel_color.y), clamp(pixel_color.z)) * 0.25;
            }

            //...indicador de progreso (actualizado cada 5 líneas)
            if(y % 5 == 0) {
                std::cerr << "\rRendering (" << samps * 1 << " spp) " << 
                             int(100.0 * y / (h - 1)) << "% completed" << std::flush;
            }
        }

        //...eescribir la imagen en un archivo PPM
        std::ofstream ofs("tuimagen.ppm", std::ios::out | std::ios::binary);
        ofs << "P3\n" << w << " " << h << "\n255\n";
        for(auto &color : c) {
            ofs << toInt(color.x) << " " << toInt(color.y) << " " << toInt(color.z) << " ";
        }
        ofs.close();
        std::cerr << "\nRenderizado completado. Imagen guardada como 'tuimagen.ppm'.\n";
    }

} //...fin del namespace PathTracer

//...mi punto de entrada
int main(int argc, char *argv[]) {
    PathTracer::render(argc, argv);
    return 0;
}
//............................................| CODE END |............................................
