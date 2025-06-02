#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "basic_camera.h"
#include "camera.h"
#include "pointLight.h"
#include "stb_image.h"
#include "cube.h"
#include "SpotLight.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shader.h"
#include "basic_camera.h"
#include <iostream>
#include <vector>
#include <cmath>
#include "pointLight.h"

#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>

#include <iostream>
#include <filesystem>
#include <cstdlib> // For rand()
#include <ctime>   // For time()


#include <iostream>


#define M_PI 3.14159265358979323846
// Example to manage smoothing groups
struct SmoothingGroup {
    std::vector<int> faces; // Indices of faces in this group
};

std::unordered_map<int, SmoothingGroup> smoothingGroups;

void parseOBJ(const std::string& filePath,
    std::vector<glm::vec3>& vertices,
    std::vector<std::vector<int>>& faces) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return;
    }

    int currentSmoothingGroup = 0;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string prefix;
        ss >> prefix;

        if (prefix == "v") {
            glm::vec3 vertex;
            ss >> vertex.x >> vertex.y >> vertex.z;
            vertices.push_back(vertex);
        }
        else if (prefix == "f") {
            std::vector<int> face;
            std::string vertexStr;

            while (ss >> vertexStr) {
                size_t doubleSlashPos = vertexStr.find("//");
                if (doubleSlashPos != std::string::npos) {
                    // Handle "f 1//1 2//2 3//3"
                    std::string vertexIndex = vertexStr.substr(0, doubleSlashPos);
                    if (!vertexIndex.empty()) {
                        int index = std::stoi(vertexIndex) - 1; // Convert to 0-based index
                        face.push_back(index);
                    }
                }
                else if (vertexStr.find("/") == std::string::npos) {
                    // Handle "f 1 2 3"
                    int index = std::stoi(vertexStr) - 1; // Convert to 0-based index
                    face.push_back(index);
                }
                else {
                    std::cerr << "Unexpected face format: " << vertexStr << std::endl;
                }
            }

            if (face.size() >= 3) {
                faces.push_back(face);

                // Add face to current smoothing group
                if (currentSmoothingGroup > 0) {
                    smoothingGroups[currentSmoothingGroup].faces.push_back(faces.size() - 1);
                }
            }
        }
        else if (prefix == "s") {
            std::string group;
            ss >> group;
            if (group == "off" || group == "0") {
                currentSmoothingGroup = 0;
            }
            else {
                currentSmoothingGroup = std::stoi(group);
            }
        }
    }

   

    file.close();
   
}






// Compute normals and create combined array
std::vector<float> computeNormals(const std::vector<glm::vec3>& vertices, const std::vector<std::vector<int>>& faces) {
    std::vector<glm::vec3> normals(vertices.size(), glm::vec3(0.0f));
    std::vector<float> vertexData;

    // Calculate face normals
    for (const auto& face : faces) {
        glm::vec3 v0 = vertices[face[0]];
        glm::vec3 v1 = vertices[face[1]];
        glm::vec3 v2 = vertices[face[2]];

        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));

        // Add face normal to vertex normals (for smooth shading)
        for (int index : face) {
            normals[index] += faceNormal;
        }
    }

    // Normalize vertex normals
    for (auto& normal : normals) {
        normal = glm::normalize(normal);
    }

    // Combine positions and normals
    for (size_t i = 0; i < vertices.size(); ++i) {
        const glm::vec3& position = vertices[i];
        const glm::vec3& normal = normals[i];

        vertexData.push_back(position.x);
        vertexData.push_back(position.y);
        vertexData.push_back(position.z);
        vertexData.push_back(normal.x);
        vertexData.push_back(normal.y);
        vertexData.push_back(normal.z);
    }

    return vertexData;
}

glm::vec2 calculateBoxUV(const glm::vec3& vertex) {
    glm::vec3 absVertex = glm::abs(vertex);
    if (absVertex.x > absVertex.y && absVertex.x > absVertex.z) {
        return glm::vec2(vertex.y, vertex.z); // Project onto YZ plane
    }
    else if (absVertex.y > absVertex.z) {
        return glm::vec2(vertex.x, vertex.z); // Project onto XZ plane
    }
    else {
        return glm::vec2(vertex.x, vertex.y); // Project onto XY plane
    }
}

glm::vec2 calculateSphericalUV(const glm::vec3& vertex) {
    float theta = atan2(vertex.z, vertex.x); // Angle around the Y-axis
    float phi = acos(vertex.y / glm::length(vertex)); // Angle from the Y-axis

    float u = (theta + M_PI) / (2.0f * M_PI); // Normalize to [0, 1]
    float v = phi / M_PI; // Normalize to [0, 1]

    return glm::vec2(u, v);
}


glm::vec2 calculatePlanarUV(const glm::vec3& vertex) {
    return glm::vec2(vertex.x, vertex.z); // Map x and z to u and v
}


std::vector<float> computeNormalsAndGeneratedUV(const std::vector<glm::vec3>& vertices,
    const std::vector<std::vector<int>>& faces) {
    std::vector<glm::vec3> normals(vertices.size(), glm::vec3(0.0f));
    std::vector<float> vertexData;

    // Calculate face normals
    for (const auto& face : faces) {
        glm::vec3 v0 = vertices[face[0]];
        glm::vec3 v1 = vertices[face[1]];
        glm::vec3 v2 = vertices[face[2]];

        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));

        for (int index : face) {
            normals[index] += faceNormal;
        }
    }

    // Normalize vertex normals
    for (auto& normal : normals) {
        normal = glm::normalize(normal);
    }

    // Combine positions, normals, and generated UVs
    for (size_t i = 0; i < vertices.size(); ++i) {
        const glm::vec3& position = vertices[i];
        const glm::vec3& normal = normals[i];
        glm::vec2 uv = calculateSphericalUV(position); // Or use calculateSphericalUV/calculateBoxUV

        vertexData.push_back(position.x);
        vertexData.push_back(position.y);
        vertexData.push_back(position.z);
        vertexData.push_back(normal.x);
        vertexData.push_back(normal.y);
        vertexData.push_back(normal.z);
        vertexData.push_back(uv.x);
        vertexData.push_back(uv.y);
    }

    return vertexData;
}




void generateSphere(float radius, int sectorCount, int stackCount, std::vector<float>& vertices, std::vector<unsigned int>& indices) {
    float x, y, z, xy;                              // vertex position
    float nx, ny, nz, lengthInv = 1.0f / radius;    // vertex normal, lenginv is the inverse of the radius


    float sectorStep = 2 * M_PI / sectorCount;
    float stackStep = M_PI / stackCount;
    float sectorAngle, stackAngle;

    for (int i = 0; i <= stackCount; ++i) {
        stackAngle = M_PI / 2 - i * stackStep;        // starting from pi/2 to -pi/2
        xy = radius * cosf(stackAngle);             // r * cos(u)
        z = radius * sinf(stackAngle);              // r * sin(u)

        for (int j = 0; j <= sectorCount; ++j) {
            sectorAngle = j * sectorStep;           // starting from 0 to 2pi

            // vertex position (x, y, z)
            x = xy * cosf(sectorAngle);             // r * cos(u) * cos(v)
            y = xy * sinf(sectorAngle);             // r * cos(u) * sin(v)
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);

            // normalized vertex normal (nx, ny, nz)
            nx = x * lengthInv;
            ny = y * lengthInv;
            nz = z * lengthInv;
            vertices.push_back(nx);
            vertices.push_back(ny);
            vertices.push_back(nz);
        }
    }

    // generate indices
    int k1, k2;
    for (int i = 0; i < stackCount; ++i) {
        k1 = i * (sectorCount + 1);     // beginning of current stack
        k2 = k1 + sectorCount + 1;      // beginning of next stack

        for (int j = 0; j < sectorCount; ++j, ++k1, ++k2) {
            // 2 triangles per sector excluding first and last stacks
            if (i != 0) {
                indices.push_back(k1);
                indices.push_back(k2);
                indices.push_back(k1 + 1);
            }

            if (i != (stackCount - 1)) {
                indices.push_back(k1 + 1);
                indices.push_back(k2);
                indices.push_back(k2 + 1);
            }
        }
    }
}











// Function to generate fractal tree branches with normals in 3D
void generateTree(std::vector<float>& vertices, std::vector<unsigned int>& indices, glm::vec3 start, glm::vec3 direction, float length, int depth, int maxDepth) {
    if (depth == 0 || length < 0.01f) return;

    // Calculate the end point of the current branch
    glm::vec3 end = start + direction * length;
    glm::vec3 normal = glm::normalize(glm::cross(direction, glm::vec3(0.0f, 0.0f, 1.0f)));

    // Add vertices and normals for the branch
    unsigned int startIndex = vertices.size() / 6;
    vertices.push_back(start.x);
    vertices.push_back(start.y);
    vertices.push_back(start.z);
    vertices.push_back(normal.x);
    vertices.push_back(normal.y);
    vertices.push_back(normal.z);

    vertices.push_back(end.x);
    vertices.push_back(end.y);
    vertices.push_back(end.z);
    vertices.push_back(normal.x);
    vertices.push_back(normal.y);
    vertices.push_back(normal.z);

    // Add indices for the branch
    indices.push_back(startIndex);
    indices.push_back(startIndex + 1);

    // Recursive branching (left, right, and up branches)
    float angleOffset = glm::radians(30.0f + (rand() % 10 - 5)); // Randomize angle slightly
    int numBranches = 3; // Three branches: left, right, and up

    for (int i = 0; i < numBranches; ++i) {
        glm::vec3 newDirection = direction; // Start with the current direction

        // Apply rotation based on the branch index
        if (i == 0) {
            glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), angleOffset, glm::vec3(1.0f, 0.0f, 0.0f)); // Rotate around x-axis
            newDirection = glm::vec3(rotationMatrix * glm::vec4(direction, 0.0f));
        }
        else if (i == 1) {
            glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), angleOffset, glm::vec3(0.0f, 1.0f, 0.0f)); // Rotate around y-axis
            newDirection = glm::vec3(rotationMatrix * glm::vec4(direction, 0.0f));
        }
        else {
            glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), angleOffset, glm::vec3(0.0f, 0.0f, 1.0f)); // Rotate around z-axis
            newDirection = glm::vec3(rotationMatrix * glm::vec4(direction, 0.0f));
        }

        // Random length factor to ensure it doesn't go to zero
        float lengthFactor = 0.6f + static_cast<float>(rand()) / RAND_MAX * 0.2f; // Random length factor
        generateTree(vertices, indices, end, newDirection, length * lengthFactor, depth - 1, maxDepth);
    }
}





#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


using namespace std;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
void lightEffect(unsigned int VAO, Shader lightShader, glm::mat4 model, glm::vec3 color);
void drawCube(unsigned int VAO, Shader shader, glm::mat4 model, glm::vec4 color);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void BezierCurve(double t, float xy[2], GLfloat ctrlpoints[], int L);
void read_file(string file_name, vector<float>& vec);
unsigned int hollowBezier(GLfloat ctrlpoints[], int L, vector<float>& coordinates, vector<float>& normals, vector<int>& indices, vector<float>& vertices, float div);
unsigned int loadTexture(char const* path, GLenum textureWrappingModeS, GLenum textureWrappingModeT, GLenum textureFilteringModeMin, GLenum textureFilteringModeMax);
long long nCr(int n, int r);
void load_texture(unsigned int& texture, string image_name, GLenum format);
void drawMissile(Shader& shader, unsigned int texture, unsigned int ropeVAO, unsigned int VAO_P, const std::vector<int>& indices, const glm::mat4& translationMatrix);

// draw object functions
//void drawCube(Shader shaderProgram, unsigned int VAO, glm::mat4 parentTrans, float posX = 0.0, float posY = 0.0, float posz = 0.0, float rotX = 0.0, float rotY = 0.0, float rotZ = 0.0, float scX = 1.0, float scY = 1.0, float scZ = 1.0);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// modelling transform
float rotateAngle_X = 45.0;
float rotateAngle_Y = 45.0;
float rotateAngle_Z = 45.0;
float rotateAxis_X = 0.0;
float rotateAxis_Y = 0.0;
float rotateAxis_Z = 1.0;
float translate_X = 0.0;
float translate_Y = 0.0;
float translate_Z = 0.0;
float scale_X = 1.0;
float scale_Y = 1.0;
float scale_Z = 1.0;

//// camera
//float lastX = SCR_WIDTH / 2.0f;
//float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// camera
Camera camera(glm::vec3(2.0f, 5.0f, 10.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;

float eyeX = 1.35, eyeY = 4.8, eyeZ = 10.0;
float lookAtX = 4.0, lookAtY = 4.0, lookAtZ = 6.0;
glm::vec3 V = glm::vec3(0.0f, 1.0f, 0.0f);
BasicCamera basic_camera(eyeX, eyeY, eyeZ, lookAtX, lookAtY, lookAtZ, V);

// timing
float deltaTime = 0.0f;    // time between current frame and last frame
float lastFrame = 0.0f;

bool on = false;

//birds eye
bool birdEye = false;
glm::vec3 cameraPos(-2.0f, 5.0f, 13.0f);
glm::vec3 target(-2.0f, 0.0f, 5.5f);
float birdEyeSpeed = 1.0f;


// positions of the point lights
glm::vec3 pointLightPositions[] = {
    glm::vec3(-6.0f,  1.5f,  -5.0f),
    glm::vec3(0.0f,  1.5f,  -5.0f),
    glm::vec3(-2.5f,  1.5f,  -15.0f)
    
};

PointLight pointlight1(

    pointLightPositions[0].x, pointLightPositions[0].y, pointLightPositions[0].z,  // position
    0.05f, 0.05f, 0.05f,     // ambient
    1.0f, 1.0f, 1.0f,     // diffuse
    1.0f, 1.0f, 1.0f,        // specular
    1.0f,   //k_c
    0.09f,  //k_l
    0.032f, //k_q
    1       // light number
);
PointLight pointlight2(

    pointLightPositions[1].x, pointLightPositions[1].y, pointLightPositions[1].z,  // position
    0.05f, 0.05f, 0.05f,     // ambient
    1.0f, 1.0f, 1.0f,     // diffuse
    1.0f, 1.0f, 1.0f,        // specular
    1.0f,   //k_c
    0.09f,  //k_l
    0.032f, //k_q
    2       // light number
);

PointLight pointlight3(

    pointLightPositions[2].x, pointLightPositions[2].y, pointLightPositions[2].z,  // position
    0.05f, 0.05f, 0.05f,     // ambient
    1.0f, 1.0f, 1.0f,     // diffuse
    1.0f, 1.0f, 1.0f,        // specular
    1.0f,   //k_c
    0.09f,  //k_l
    0.032f, //k_q
    3       // light number
);

SpotLight spotlight1(
    6.5f, 3.5f, 6.0f,  // position
    1.0f, 1.0f, 1.0f,     // ambient
    1.0f, 1.0f, 1.0f,      // diffuse
    1.0f, 1.0f, 1.0f,        // specular
    1.0f,   //k_c
    0.09f,  //k_l
    0.032f, //k_q
    1,       // light number
    glm::cos(glm::radians(20.5f)),
    glm::cos(glm::radians(25.5f)),
    0, -1, 0
);

SpotLight spotlight2(
    -12.5f, 3.5f, 6.0f,  // position
    1.0f, 1.0f, 1.0f,     // ambient
    1.0f, 1.0f, 1.0f,      // diffuse
    1.0f, 1.0f, 1.0f,        // specular
    1.0f,   //k_c
    0.09f,  //k_l
    0.032f, //k_q
    2,       // light number
    glm::cos(glm::radians(20.5f)),
    glm::cos(glm::radians(25.5f)),
    0, -1, 0
);




// light settings
bool onOffToggle = true;
bool ambientToggle = true;
bool diffuseToggle = true;
bool specularToggle = true;
bool dl = true;
bool spt = true;
bool point1 = true;
bool point2 = true;
bool point3 = true;



//float d_amb_on = 1.0f;
//float d_def_on = 1.0f;
//float d_spec_on = 1.0f;

glm::vec3 amb(0.2f, 0.2f, 0.2f);
glm::vec3 def(0.8f, 0.8f, 0.8f);
glm::vec3 spec(1.0f, 1.0f, 1.0f);

float fov = glm::radians(camera.Zoom);
float aspect = (float)SCR_WIDTH / (float)SCR_HEIGHT;
float near = 0.1f;
float far = 100.0f;

float tanHalfFOV = tan(fov / 2.0f);

const double pi = 3.14159265389;
const int nt = 40;
const int ntheta = 30;
unsigned int bezierVAO, ropeVAO, rotorVAO, sliderVAO, carousalVAO, headVAO, hillVAO,pillarVAO;

vector <float> cntrlPoints, cntrlPointsRope, cntrlPointsRope2, cntrlPointsRotor, cntrlPointsCylinder, cntrlPointsCarousal, cntrlPointsHead, cntrlPointsPilar;
vector <float> coordinates, coordinatesRope, coordinatesRope2,coordinatesRotor, coordinatesCylinder, coordinatesCarousal, coordinatesHead, coordinatesPillar;
vector <float> normals, normalsRope, normalsRope2, normalsRotor, normalsCylinder, normalsCarousal, normalsHead, normalsPillar;
vector <int> indices, indicesRope, indicesRope2,indicesRotor, indicesCylinder, indicesCarousal, indicesHead, indicesPillar;
vector <float> vertices, verticesRope, verticesRotor, verticesRope2, verticesCylinder, verticesCarousal, verticesHead, verticesPillar;
//vector <float> textureCoords, textureCoordsRope, textureCoordsRope2;

// texture
float extra = 4.0f;
float TXmin = 0.0f;
float TXmax = 1.0f ;
float TYmin = 0.0f;
float TYmax = 1.0f ;

//doors
bool openDoor = true;
float doorAngle =90.0f;

bool texture_bool = true;

float rightGunRotationAngle = 0.0f; // Initial rotation angle for the third object
float rightBaseRotationAngle = 0.0f; // Initial rotation angle for both objects

bool rightGunRotation = false;
bool rightBaseRotation = false;




float leftBaseRotationAngle = 0.0f; // Initial rotation angle for both objects
float leftGunRotationAngle = 0.0f;



float t = 0.0f; // Initial parameter for the parabolic path
float tSpeed = 0.3f; // Speed of the parameter change


// Diagonal translation speed
float translationSpeed = 3.0f;
glm::vec3 objectPosition1 = glm::vec3(-6.4, -0.45, 2.7); // Initial position
glm::vec3 objectPosition2 = glm::vec3(-6.7, -0.45, 2.75);
glm::vec3 objectPosition3 = glm::vec3(-6.65, -0.47, 3.0);
glm::vec3 objectPosition4 = glm::vec3(-6.4, -0.47, 3.0);


// Function to draw a sphere
void drawSphere(unsigned int& VAO_S, Shader& lightingShader, glm::vec3 color, glm::mat4 model, std::vector<unsigned int>& indices)
{
    lightingShader.use();

    // Setting up materialistic property
    lightingShader.setVec3("material.ambient", color);
    lightingShader.setVec3("material.diffuse", color);
    lightingShader.setVec3("material.specular", color);
    lightingShader.setFloat("material.shininess", 32.0f);
    //float emissiveIntensity = 0.05f; // Adjust this value to increase or decrease the intensity
    //glm::vec3 emissiveColor = glm::vec3(1.0f, 0.0f, 0.0f) * emissiveIntensity;

    //lightingShader.setVec3("material.emissive", emissiveColor);

    lightingShader.setMat4("model", model);

    glBindVertexArray(VAO_S);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}



int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "CSE 4208: Computer Graphics Laboratory", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetKeyCallback(window, key_callback);
    //glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile our shader zprogram
    // ------------------------------------
    Shader ourShader("vertexShader.vs", "fragmentShader.fs");

    //Shader constantShader("vertexShader.vs", "fragmentShaderV2.fs");

    //Shader lightingShader("vertexShaderForGouraudShading.vs", "fragmentShaderForGouraudShading.fs");
    Shader lightingShaderWithTexture("vertexShaderForPhongShadingWithTexture.vs", "fragmentShaderForPhongShadingWithTexture.fs");
    Shader textureShader("texture_vertex.vs", "texture_fragment.fs");
    Shader lightingShader("vertexShaderForPhongShading.vs", "fragmentShaderForPhongShading.fs");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    /*float cube_vertices[] = {
        // positions      // normals
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f,
        1.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f,

        1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,

        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,

        0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 1.0f, -1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,

        1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f,

        0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f,
        1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f
    };*/
    /*unsigned int cube_indices[] = {
        0, 3, 2,
        2, 1, 0,

        4, 5, 7,
        7, 6, 4,

        8, 9, 10,
        10, 11, 8,

        12, 13, 14,
        14, 15, 12,

        16, 17, 18,
        18, 19, 16,

        20, 21, 22,
        22, 23, 20
    };*/

    float cube_vertices[] = {
        // positions      // normals         // texture coords
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, TXmax, TYmin,
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, TXmin, TYmin,
        1.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, TXmin, TYmax,
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, TXmax, TYmax,

        1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, TXmax, TYmin,
        1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, TXmax, TYmax,
        1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, TXmin, TYmin,
        1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, TXmin, TYmax,

        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, TXmin, TYmin,
        1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, TXmax, TYmin,
        1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, TXmax, TYmax,
        0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, TXmin, TYmax,

        0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, TXmax, TYmin,
        0.0f, 1.0f, 1.0f, -1.0f, 0.0f, 0.0f, TXmax, TYmax,
        0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, TXmin, TYmax,
        0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, TXmin, TYmin,

        1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, TXmax, TYmin,
        1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, TXmax, TYmax,
        0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, TXmin, TYmax,
        0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, TXmin, TYmin,

        0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, TXmin, TYmin,
        1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, TXmax, TYmin,
        1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, TXmax, TYmax,
        0.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, TXmin, TYmax
    };
    unsigned int cube_indices[] = {
        0, 3, 2,
        2, 1, 0,

        4, 5, 7,
        7, 6, 4,

        8, 9, 10,
        10, 11, 8,

        12, 13, 14,
        14, 15, 12,

        16, 17, 18,
        18, 19, 16,

        20, 21, 22,
        22, 23, 20
    };
    unsigned int cubeVAO, cubeVBO, cubeEBO;
    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO);
    glGenBuffers(1, &cubeEBO);

    glBindVertexArray(cubeVAO);

    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices), cube_indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    //vertex normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)12);
    glEnableVertexAttribArray(1);

    // texture attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)24);
    glEnableVertexAttribArray(2);

    //light's VAO
    unsigned int lightCubeVAO;
    glGenVertexArrays(1, &lightCubeVAO);
    glBindVertexArray(lightCubeVAO);

    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);


    float prism_vertices[] = {
        // Triangular End 1 (e.g., left end)
        // Positions        // Normals             // Texture Coords
        0.0f, 0.0f, 0.0f,   0.0f,  0.0f, -1.0f,   TXmin, TYmin, // Vertex 0
        1.0f, 0.0f, 0.0f,   0.0f,  0.0f, -1.0f,   TXmax, TYmin, // Vertex 1
        0.0f, 1.0f, 0.0f,   0.0f,  0.0f, -1.0f,   TXmin, TYmax, // Vertex 2

        // Triangular End 2 (e.g., right end)
        // Positions        // Normals             // Texture Coords
        0.0f, 0.0f, 1.0f,   0.0f,  0.0f, 1.0f,     TXmin, TYmin, // Vertex 3
        1.0f, 0.0f, 1.0f,   0.0f,  0.0f, 1.0f,     TXmax, TYmin, // Vertex 4
        0.0f, 1.0f, 1.0f,   0.0f,  0.0f, 1.0f,     TXmin, TYmax, // Vertex 5

        // Rectangular Side 1 (Bottom Face)
        // Positions        // Normals             // Texture Coords
        0.0f, 0.0f, 0.0f,   0.0f, -1.0f, 0.0f,     TXmin, TYmin, // Vertex 6
        1.0f, 0.0f, 0.0f,   0.0f, -1.0f, 0.0f,     TXmax, TYmin, // Vertex 7
        1.0f, 0.0f, 1.0f,   0.0f, -1.0f, 0.0f,     TXmax, TYmax, // Vertex 8
        0.0f, 0.0f, 1.0f,   0.0f, -1.0f, 0.0f,     TXmin, TYmax, // Vertex 9

        // Rectangular Side 2 (Vertical Face Adjacent to Y-axis)
        // Positions        // Normals             // Texture Coords
        0.0f, 0.0f, 0.0f,  -1.0f,  0.0f, 0.0f,     TXmax, TYmin, // Vertex 10
        0.0f, 1.0f, 0.0f,  -1.0f,  0.0f, 0.0f,     TXmax, TYmax, // Vertex 11
        0.0f, 1.0f, 1.0f,  -1.0f,  0.0f, 0.0f,     TXmin, TYmax, // Vertex 12
        0.0f, 0.0f, 1.0f,  -1.0f,  0.0f, 0.0f,     TXmin, TYmin, // Vertex 13

        // Rectangular Side 3 (Hypotenuse Face)
        // Positions        // Normals             // Texture Coords
        0.0f, 1.0f, 0.0f,   0.0f,  1.0f, 0.0f,     TXmax, TYmin, // Vertex 14
        1.0f, 0.0f, 0.0f,   0.0f,  1.0f, 0.0f,     TXmax, TYmax, // Vertex 15
        1.0f, 0.0f, 1.0f,   0.0f,  1.0f, 0.0f,     TXmin, TYmax, // Vertex 16
        0.0f, 1.0f, 1.0f,   0.0f,  1.0f, 0.0f,     TXmin, TYmin, // Vertex 17
    };

    unsigned int prism_indices[] = {
        // Triangular End 1
        0, 1, 2,

        // Triangular End 2
        3, 5, 4,

        // Rectangular Side 1 (Bottom Face)
        6, 7, 8,
        8, 9, 6,

        // Rectangular Side 2 (Vertical Face Adjacent to Y-axis)
        10, 11, 12,
        12, 13, 10,

        // Rectangular Side 3 (Hypotenuse Face)
        14, 15, 16,
        16, 17, 14
    };

    unsigned int VAO_P, VBO_P, EBO_P;
    glGenVertexArrays(1, &VAO_P);
    glGenBuffers(1, &VBO_P);
    glGenBuffers(1, &EBO_P);

    glBindVertexArray(VAO_P);

    // Vertex Buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO_P);
    glBufferData(GL_ARRAY_BUFFER, sizeof(prism_vertices), prism_vertices, GL_STATIC_DRAW);

    // Element Buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_P);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(prism_indices), prism_indices, GL_STATIC_DRAW);

    // Position Attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal Attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Texture Coord Attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // Unbind VAO
    glBindVertexArray(0);


    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    //ourShader.use();
    //constantShader.use();

    std::vector<glm::vec3> tvertices;
    std::vector<std::vector<int>> tfaces;

    // Load OBJ file
    parseOBJ("my.txt", tvertices, tfaces);

    // Compute normals and create vertex data array
    std::vector<float> vertexData = computeNormals(tvertices, tfaces);



    // Flatten faces into a single indices vector
    std::vector<unsigned int> tindices;
    for (const auto& face : tfaces) {
        for (int index : face) {
            tindices.push_back(static_cast<unsigned int>(index)); // Corrected
        }
    }

    // OpenGL Buffer Setup
    unsigned int VBOt, VAOt, EBOt;
    glGenVertexArrays(1, &VAOt);
    glGenBuffers(1, &VBOt);
    glGenBuffers(1, &EBOt);

    glBindVertexArray(VAOt);

    // VBO for vertex data (positions and normals are precomputed in vertexData)
    glBindBuffer(GL_ARRAY_BUFFER, VBOt);
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_STATIC_DRAW);

    // EBO for face indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBOt);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, tindices.size() * sizeof(unsigned int), tindices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind VAO (optional for safety)
    glBindVertexArray(0);
    

	//------------------------------------------------------------------texture test-------------

    std::vector<glm::vec3> tt_vertices;
    std::vector<std::vector<int>> tt_faces;

    // Load OBJ file
    parseOBJ("my.txt", tt_vertices, tt_faces);

    // Compute normals and create vertex data array
    std::vector<float> t_vertexData = computeNormalsAndGeneratedUV(tt_vertices, tt_faces);



    // Flatten faces into a single indices vector
    std::vector<unsigned int> tt_indices;
    for (const auto& face : tt_faces) {
        for (int index : face) {
            tt_indices.push_back(static_cast<unsigned int>(index)); // Corrected
        }
    }

    // OpenGL Buffer Setup
    unsigned int VBOtt, VAOtt, EBOtt;
    glGenVertexArrays(1, &VAOtt);
    glGenBuffers(1, &VBOtt);
    glGenBuffers(1, &EBOtt);

    glBindVertexArray(VAOtt);

    // VBO for vertex data (positions and normals are precomputed in vertexData)
    glBindBuffer(GL_ARRAY_BUFFER, VBOtt);
    glBufferData(GL_ARRAY_BUFFER, t_vertexData.size() * sizeof(float), t_vertexData.data(), GL_STATIC_DRAW);

    // EBO for face indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBOtt);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, tt_indices.size() * sizeof(unsigned int), tt_indices.data(), GL_STATIC_DRAW);

    // Position attribute: 3 floats (x, y, z)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute: 3 floats (nx, ny, nz)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Texture Coord Attribute: 2 floats (u, v)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // Unbind VAO
    glBindVertexArray(0);

    // Unbind VAO (optional for safety)
    glBindVertexArray(0);






    // Generate sphere data
    std::vector<float> vertices_s;
    std::vector<unsigned int> indices_s;
    generateSphere(1.0f, 72, 72, vertices_s, indices_s);

    // Create VAO_S, VBO_S, and EBO_S
    unsigned int VAO_S, VBO_S, EBO_S;
    glGenVertexArrays(1, &VAO_S);
    glGenBuffers(1, &VBO_S);
    glGenBuffers(1, &EBO_S);

    // Bind VAO
    glBindVertexArray(VAO_S);

    // Bind and set VBO
    glBindBuffer(GL_ARRAY_BUFFER, VBO_S);
    glBufferData(GL_ARRAY_BUFFER, vertices_s.size() * sizeof(float), vertices_s.data(), GL_STATIC_DRAW);

    // Bind and set EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_S);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_s.size() * sizeof(unsigned int), indices_s.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind VAO
    glBindVertexArray(0);





	//------------------------------------------------------------------

    std::vector<glm::vec3> rbvertices;
    std::vector<std::vector<int>> rbfaces;

    // Load OBJ file
    parseOBJ("ram_main.txt", rbvertices, rbfaces);

    // Compute normals and create vertex data array
    std::vector<float> rbvertexData = computeNormals(rbvertices, rbfaces);



    // Flatten faces into a single indices vector
    std::vector<unsigned int> rbindices;
    for (const auto& face : rbfaces) {
        for (int index : face) {
            rbindices.push_back(static_cast<unsigned int>(index)); // Corrected
        }
    }

    // OpenGL Buffer Setup
    unsigned int VBOrb, VAOrb, EBOrb;
    glGenVertexArrays(1, &VAOrb);
    glGenBuffers(1, &VBOrb);
    glGenBuffers(1, &EBOrb);

    glBindVertexArray(VAOrb);

    // VBO for vertex data (positions and normals are precomputed in vertexData)
    glBindBuffer(GL_ARRAY_BUFFER, VBOrb);
    glBufferData(GL_ARRAY_BUFFER, rbvertexData.size() * sizeof(float), rbvertexData.data(), GL_STATIC_DRAW);

    // EBO for face indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBOrb);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, rbindices.size() * sizeof(unsigned int), rbindices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind VAO (optional for safety)
    glBindVertexArray(0);

   


    //------------------------------------------------------------------

    std::vector<glm::vec3> rbbvertices;
    std::vector<std::vector<int>> rbbfaces;

    // Load OBJ file
    parseOBJ("ram_base.txt", rbbvertices, rbbfaces);

    // Compute normals and create vertex data array
    std::vector<float> rbbvertexData = computeNormals(rbbvertices, rbbfaces);



    // Flatten faces into a single indices vector
    std::vector<unsigned int> rbbindices;
    for (const auto& face : rbbfaces) {
        for (int index : face) {
            rbbindices.push_back(static_cast<unsigned int>(index)); // Corrected
        }
    }

    // OpenGL Buffer Setup
    unsigned int VBOrbb, VAOrbb, EBOrbb;
    glGenVertexArrays(1, &VAOrbb);
    glGenBuffers(1, &VBOrbb);
    glGenBuffers(1, &EBOrbb);

    glBindVertexArray(VAOrbb);

    // VBO for vertex data (positions and normals are precomputed in vertexData)
    glBindBuffer(GL_ARRAY_BUFFER, VBOrbb);
    glBufferData(GL_ARRAY_BUFFER, rbbvertexData.size() * sizeof(float), rbbvertexData.data(), GL_STATIC_DRAW);

    // EBO for face indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBOrbb);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, rbbindices.size() * sizeof(unsigned int), rbbindices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind VAO (optional for safety)
    glBindVertexArray(0);

    //------------------------------------------------------------------

    std::vector<glm::vec3> rbgvertices;
    std::vector<std::vector<int>> rbgfaces;

    // Load OBJ file
    parseOBJ("ram_gun.txt", rbgvertices, rbgfaces);

    // Compute normals and create vertex data array
    std::vector<float> rbgvertexData = computeNormals(rbgvertices, rbgfaces);



    // Flatten faces into a single indices vector
    std::vector<unsigned int> rbgindices;
    for (const auto& face : rbgfaces) {
        for (int index : face) {
            rbgindices.push_back(static_cast<unsigned int>(index)); // Corrected
        }
    }

    // OpenGL Buffer Setup
    unsigned int VBOrbg, VAOrbg, EBOrbg;
    glGenVertexArrays(1, &VAOrbg);
    glGenBuffers(1, &VBOrbg);
    glGenBuffers(1, &EBOrbg);

    glBindVertexArray(VAOrbg);

    // VBO for vertex data (positions and normals are precomputed in vertexData)
    glBindBuffer(GL_ARRAY_BUFFER, VBOrbg);
    glBufferData(GL_ARRAY_BUFFER, rbgvertexData.size() * sizeof(float), rbgvertexData.data(), GL_STATIC_DRAW);

    // EBO for face indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBOrbg);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, rbgindices.size() * sizeof(unsigned int), rbgindices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind VAO (optional for safety)
    glBindVertexArray(0);


    //-------------------------------------------pot-------------------
    std::vector<glm::vec3> potvertices;
    std::vector<std::vector<int>> potfaces;

    // Load OBJ file
    parseOBJ("pot.txt", potvertices, potfaces);

    // Compute normals and create vertex data array
    std::vector<float> potvertexData = computeNormalsAndGeneratedUV(potvertices, potfaces);



    // Flatten faces into a single indices vector
    std::vector<unsigned int> potindices;
    for (const auto& face : potfaces) {
        for (int index : face) {
            potindices.push_back(static_cast<unsigned int>(index)); // Corrected
        }
    }

    // OpenGL Buffer Setup
    unsigned int VBOpot, VAOpot, EBOpot;
    glGenVertexArrays(1, &VAOpot);
    glGenBuffers(1, &VBOpot);
    glGenBuffers(1, &EBOpot);

    glBindVertexArray(VAOpot);

    // VBO for vertex data (positions and normals are precomputed in vertexData)
    glBindBuffer(GL_ARRAY_BUFFER, VBOpot);
    glBufferData(GL_ARRAY_BUFFER, potvertexData.size() * sizeof(float), potvertexData.data(), GL_STATIC_DRAW);

    // EBO for face indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBOpot);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, potindices.size() * sizeof(unsigned int), potindices.data(), GL_STATIC_DRAW);

    // Position attribute: 3 floats (x, y, z)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute: 3 floats (nx, ny, nz)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Texture Coord Attribute: 2 floats (u, v)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // Unbind VAO
    glBindVertexArray(0);

    // Unbind VAO (optional for safety)
    glBindVertexArray(0);

    // Unbind VAO (optional for safety)
    glBindVertexArray(0);


    //----------------------------solid tree---------------------
    //------------------------------------------------------------------

    std::vector<glm::vec3> trvertices;
    std::vector<std::vector<int>> trfaces;

    // Load OBJ file
    parseOBJ("tree.txt", trvertices, trfaces);

    // Compute normals and create vertex data array
    std::vector<float> trvertexData = computeNormals(trvertices, trfaces);



    // Flatten faces into a single indices vector
    std::vector<unsigned int> trindices;
    for (const auto& face : trfaces) {
        for (int index : face) {
            trindices.push_back(static_cast<unsigned int>(index)); // Corrected
        }
    }

    // OpenGL Buffer Setup
    unsigned int VBOtr, VAOtr, EBOtr;
    glGenVertexArrays(1, &VAOtr);
    glGenBuffers(1, &VBOtr);
    glGenBuffers(1, &EBOtr);

    glBindVertexArray(VAOtr);

    // VBO for vertex data (positions and normals are precomputed in vertexData)
    glBindBuffer(GL_ARRAY_BUFFER, VBOtr);
    glBufferData(GL_ARRAY_BUFFER, trvertexData.size() * sizeof(float), trvertexData.data(), GL_STATIC_DRAW);

    // EBO for face indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBOtr);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, trindices.size() * sizeof(unsigned int), trindices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind VAO (optional for safety)
    glBindVertexArray(0);

    //------------------------------------------------------trunk


    std::vector<glm::vec3> trunkvertices;
    std::vector<std::vector<int>> trunkfaces;

    // Load OBJ file
    parseOBJ("trunk.txt", trunkvertices, trunkfaces);

    // Compute normals and create vertex data array
    std::vector<float> trunkvertexData = computeNormals(trunkvertices, trunkfaces);



    // Flatten faces into a single indices vector
    std::vector<unsigned int> trunkindices;
    for (const auto& face : trunkfaces) {
        for (int index : face) {
            trunkindices.push_back(static_cast<unsigned int>(index)); // Corrected
        }
    }

    // OpenGL Buffer Setup
    unsigned int VBOtrunk, VAOtrunk, EBOtrunk;
    glGenVertexArrays(1, &VAOtrunk);
    glGenBuffers(1, &VBOtrunk);
    glGenBuffers(1, &EBOtrunk);

    glBindVertexArray(VAOtrunk);

    // VBO for vertex data (positions and normals are precomputed in vertexData)
    glBindBuffer(GL_ARRAY_BUFFER, VBOtrunk);
    glBufferData(GL_ARRAY_BUFFER, trunkvertexData.size() * sizeof(float), trunkvertexData.data(), GL_STATIC_DRAW);

    // EBO for face indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBOtrunk);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, trunkindices.size() * sizeof(unsigned int), trunkindices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind VAO (optional for safety)
    glBindVertexArray(0);


    //--------------------------------------------------------------------------rocket
  


    std::vector<glm::vec3> missilevertices;
    std::vector<std::vector<int>> missilefaces;

    // Load OBJ file
    parseOBJ("missile.txt", missilevertices, missilefaces);

    // Compute normals and create vertex data array
    std::vector<float> missilevertexData = computeNormals(missilevertices, missilefaces);



    // Flatten faces into a single indices vector
    std::vector<unsigned int> missileindices;
    for (const auto& face : missilefaces) {
        for (int index : face) {
            missileindices.push_back(static_cast<unsigned int>(index)); // Corrected
        }
    }

    // OpenGL Buffer Setup
    unsigned int VBOm, VAOm, EBOm;
    glGenVertexArrays(1, &VAOm);
    glGenBuffers(1, &VBOm);
    glGenBuffers(1, &EBOm);

    glBindVertexArray(VAOm);

    // VBO for vertex data (positions and normals are precomputed in vertexData)
    glBindBuffer(GL_ARRAY_BUFFER, VBOm);
    glBufferData(GL_ARRAY_BUFFER, missilevertexData.size() * sizeof(float), missilevertexData.data(), GL_STATIC_DRAW);

    // EBO for face indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBOm);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, missileindices.size() * sizeof(unsigned int), missileindices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind VAO (optional for safety)
    glBindVertexArray(0);



    // Generate tree vertices and indices
    std::vector<float> treeVertices;
    std::vector<unsigned int> treeIndices;
    generateTree(treeVertices, treeIndices, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 1.0f, 5, 10); // Start at origin, direction up, length 1.0, depth 5

    // Create VAO, VBO, and EBO for the tree
    unsigned int treeVAO, treeVBO, treeEBO;
    glGenVertexArrays(1, &treeVAO);
    glGenBuffers(1, &treeVBO);
    glGenBuffers(1, &treeEBO);

    glBindVertexArray(treeVAO);

    glBindBuffer(GL_ARRAY_BUFFER, treeVBO);
    glBufferData(GL_ARRAY_BUFFER, treeVertices.size() * sizeof(float), &treeVertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, treeEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, treeIndices.size() * sizeof(unsigned int), &treeIndices[0], GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

  
    
    read_file("hill.txt", cntrlPoints);
    hillVAO = hollowBezier(cntrlPoints.data(), ((unsigned int)cntrlPoints.size() / 3) - 1, coordinates, normals, indices, vertices,1.0);

    read_file("rope_points.txt", cntrlPointsRope);
    ropeVAO = hollowBezier(cntrlPointsRope.data(), ((unsigned int)cntrlPointsRope.size() / 3) - 1, coordinatesRope, normalsRope, indicesRope, verticesRope,1.0);
    
    read_file("slider_points.txt", cntrlPointsCylinder);
    sliderVAO = hollowBezier(cntrlPointsCylinder.data(), ((unsigned int)cntrlPointsCylinder.size() / 3) - 1, coordinatesCylinder, normalsCylinder, indicesCylinder, verticesCylinder,4.0);

    read_file("pillar_points.txt", cntrlPointsPilar);
    pillarVAO = hollowBezier(cntrlPointsPilar.data(), ((unsigned int)cntrlPointsPilar.size() / 3) - 1, coordinatesPillar, normalsPillar, indicesPillar, verticesPillar, 1.0);


    // Texture loading

    //load_texture(texture1, "grass.jpg", GL_RGBA);
    
    unsigned int texture,texture2, texture3, texture4, texture5;
    string ImgPath = "dome.png";
    texture = loadTexture(ImgPath.c_str(), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);

    ImgPath = "tile.png";
    texture2 = loadTexture(ImgPath.c_str(), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);

    ImgPath = "wood.png";
    texture3 = loadTexture(ImgPath.c_str(), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);

    ImgPath = "wood.png";
    texture4 = loadTexture(ImgPath.c_str(), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);

    ImgPath = "mushroom3.jpg";
    texture5 = loadTexture(ImgPath.c_str(), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);

    float r = 0.0f;
    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

       
        // pass projection matrix to shader (note that in this case it could change every frame)
        //glm::mat4 projection = glm::perspective(glm::radians(basic_camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        //glm::mat4 projection = glm::ortho(-2.0f, +2.0f, -1.5f, +1.5f, 0.1f, 100.0f);
        //ourShader.setMat4("projection", projection);
        //constantShader.setMat4("projection", projection);

        //glm::mat4 projection(0.0f); // Initialize with zero matrix

        //projection[0][0] = 1.0f / (aspect * tanHalfFOV);
        //projection[1][1] = 1.0f / tanHalfFOV;
        //projection[2][2] = -(far + near) / (far - near);
        //projection[2][3] = -1.0f;
        //projection[3][2] = -(2.0f * far * near) / (far - near);

        //lightingShader.setMat4("projection", projection);

        // camera/view transformation

        glm::mat4 view;

        if (birdEye) {
            glm::vec3 up(0.0f, 1.0f, 0.0f);
            view = glm::lookAt(cameraPos, target, up);
        }
        else {
            //view = basic_camera.createViewMatrix();
            view = camera.GetViewMatrix();
        
        }

        glm::mat4 identityMatrix = glm::mat4(1.0f);
        glm::mat4 translateMatrix, rotateXMatrix, rotateYMatrix, rotateZMatrix, scaleMatrix, model, modelCentered,
            translateMatrixprev;
        
        
        

        /*textureShader.use();
        textureShader.setMat4("projection", projection);
        textureShader.setMat4("view", view);

        glActiveTexture(GL_TEXTURE0);
        translateMatrix = glm::translate(identityMatrix, glm::vec3(15.0, 0.0, -10.0));
        scaleMatrix = glm::scale(identityMatrix, glm::vec3(1.0, 1.0, 1.0));
        model = translateMatrix * scaleMatrix;
        textureShader.setMat4("model", model);
        //textureShader.setVec4("color", glm::vec4(1.0, 0.0, 0.0, 1.0));
        glBindTexture(GL_TEXTURE_2D, texture);
        glBindVertexArray(hillVAO);
        glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, (void*)0);*/

        


            lightingShaderWithTexture.use();
            lightingShaderWithTexture.setVec3("viewPos", camera.Position);
            lightingShaderWithTexture.setMat4("projection", projection);
            lightingShaderWithTexture.setMat4("view", view);

            // point light 1
            pointlight1.setUpPointLight(lightingShaderWithTexture);
            // point light 2
            pointlight2.setUpPointLight(lightingShaderWithTexture);
            // point light 3
            pointlight3.setUpPointLight(lightingShaderWithTexture);

            spotlight1.setUpspotLight(lightingShaderWithTexture);
			spotlight2.setUpspotLight(lightingShaderWithTexture);

            lightingShaderWithTexture.setVec3("directionalLight.directiaon", 0.0f, -3.0f, 0.0f);
            lightingShaderWithTexture.setVec3("directionalLight.ambient", .5f, .5f, .5f);
            lightingShaderWithTexture.setVec3("directionalLight.diffuse", .8f, .8f, .8f);
            lightingShaderWithTexture.setVec3("directionalLight.specular", 1.0f, 1.0f, 1.0f);

            lightingShaderWithTexture.setBool("directionLightOn", dl);






            // dome 1
            translateMatrix = glm::translate(identityMatrix, glm::vec3(2.0, 0.0, 0.0));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(1.0, 1.0, 1.0));
            model = translateMatrix * scaleMatrix;
            lightingShaderWithTexture.use();
            lightingShaderWithTexture.setInt("material.diffuse", 0); // Use texture unit 0 for diffuse
            lightingShaderWithTexture.setInt("material.specular", 1); // Use texture unit 1 for specular
            lightingShaderWithTexture.setFloat("material.shininess", 32.0f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE0); // Activate texture unit 0
            glBindTexture(GL_TEXTURE_2D, texture); // Bind texture to texture unit 0
            glBindVertexArray(hillVAO);
            glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, (void*)0);

            // dome 2
            translateMatrix = glm::translate(identityMatrix, glm::vec3(2.0, 0.0, -7.0));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(1.0, 1.0, 1.0));
            model = translateMatrix * scaleMatrix;
            lightingShaderWithTexture.use();
            lightingShaderWithTexture.setInt("material.diffuse", 0); // Use texture unit 1 for diffuse
            lightingShaderWithTexture.setInt("material.specular", 0); // Use texture unit 1 for specular
            lightingShaderWithTexture.setFloat("material.shininess", 32.0f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE0); // Activate texture unit 1
            glBindTexture(GL_TEXTURE_2D, texture); // Bind texture2 to texture unit 1
            glBindVertexArray(hillVAO);
            glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, (void*)0);

            //dome 3
            translateMatrix = glm::translate(identityMatrix, glm::vec3(-8.0, 0.0, -7.0));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(1.0, 1.0, 1.0));
            model = translateMatrix * scaleMatrix;
            lightingShaderWithTexture.setInt("material.diffuse", 0);
            lightingShaderWithTexture.setInt("material.specular", 1);
            lightingShaderWithTexture.setFloat("material.shininess", 32.0f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glBindVertexArray(hillVAO);
            glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, (void*)0);


            //dome 4
            translateMatrix = glm::translate(identityMatrix, glm::vec3(-8.0, 0.0, 0.0));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(1.0, 1.0, 1.0));
            model = translateMatrix * scaleMatrix;
            lightingShaderWithTexture.setInt("material.diffuse", 0);
            lightingShaderWithTexture.setInt("material.specular", 1);
            lightingShaderWithTexture.setFloat("material.shininess", 32.0f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glBindVertexArray(hillVAO);
            glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, (void*)0);


            //floor
            translateMatrix = glm::translate(identityMatrix, glm::vec3(-20.0, -2.0, -20.0));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(30.0, 0.1, 30.0));
            model = translateMatrix * scaleMatrix;
            lightingShaderWithTexture.setInt("material.diffuse", 1);
            lightingShaderWithTexture.setInt("material.specular", 1);
            lightingShaderWithTexture.setFloat("material.shininess", 32.0f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, texture2);
            glBindVertexArray(cubeVAO);
            glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, (void*)0);

            //left wall
            translateMatrix = glm::translate(identityMatrix, glm::vec3(-8.25, -2.0, -8.0));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.25, 2.25, 8.0));
            model = translateMatrix * scaleMatrix;
            lightingShaderWithTexture.setInt("material.diffuse", 0);
            lightingShaderWithTexture.setInt("material.specular", 1);
            lightingShaderWithTexture.setFloat("material.shininess", 32.0f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE);
            glBindTexture(GL_TEXTURE_2D, texture);
            glBindVertexArray(cubeVAO);
            glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, (void*)0);

            //Right wall
            translateMatrix = glm::translate(identityMatrix, glm::vec3(2.0, -2.0, -8.0));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.25, 2.25, 8.0));
            model = translateMatrix * scaleMatrix;
            lightingShaderWithTexture.setInt("material.diffuse", 0);
            lightingShaderWithTexture.setInt("material.specular", 1);
            lightingShaderWithTexture.setFloat("material.shininess", 32.0f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, texture);
            glBindVertexArray(cubeVAO);
            glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, (void*)0);


            //back wall
            translateMatrix = glm::translate(identityMatrix, glm::vec3(-8.5, -2.0, -7.0));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(10.0, 2.25, 0.25));
            model = translateMatrix * scaleMatrix;
            lightingShaderWithTexture.setInt("material.diffuse", 0);
            lightingShaderWithTexture.setInt("material.specular", 1);
            lightingShaderWithTexture.setFloat("material.shininess", 32.0f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, texture);
            glBindVertexArray(cubeVAO);
            glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, (void*)0);

            //Front wall-left
            translateMatrix = glm::translate(identityMatrix, glm::vec3(-7.5, -2.0, 0.0));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(3.0, 2.25, 0.25));
            model = translateMatrix * scaleMatrix;
            lightingShaderWithTexture.setInt("material.diffuse", 0);
            lightingShaderWithTexture.setInt("material.specular", 1);
            lightingShaderWithTexture.setFloat("material.shininess", 32.0f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, texture);
            glBindVertexArray(cubeVAO);
            glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, (void*)0);

            //Front wall-right
            translateMatrix = glm::translate(identityMatrix, glm::vec3(-1.5, -2.0, 0.0));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(3.0, 2.25, 0.25));
            model = translateMatrix * scaleMatrix;
            lightingShaderWithTexture.setInt("material.diffuse", 0);
            lightingShaderWithTexture.setInt("material.specular", 1);
            lightingShaderWithTexture.setFloat("material.shininess", 32.0f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, texture);
            glBindVertexArray(cubeVAO);
            glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, (void*)0);

            //Front wall-top
            translateMatrix = glm::translate(identityMatrix, glm::vec3(-5.5, -0.5, 0.0));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(4.0, 0.75, 0.25));
            model = translateMatrix * scaleMatrix;
            lightingShaderWithTexture.setInt("material.diffuse", 0);
            lightingShaderWithTexture.setInt("material.specular", 1);
            lightingShaderWithTexture.setFloat("material.shininess", 32.0f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, texture);
            glBindVertexArray(cubeVAO);
            glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, (void*)0);

            //left half-door
			translateMatrix = glm::translate(identityMatrix, glm::vec3(-4.5, -2.0, 0.0));
			scaleMatrix = glm::scale(identityMatrix, glm::vec3(1.5, 1.5, 0.125));
			rotateYMatrix = glm::rotate(identityMatrix, glm::radians(doorAngle), glm::vec3(0.0f, 1.0f, 0.0f));
			model = translateMatrix*rotateYMatrix* scaleMatrix;
            lightingShaderWithTexture.setInt("material.diffuse", 0);
            lightingShaderWithTexture.setInt("material.specular", 1);
            lightingShaderWithTexture.setFloat("material.shininess", 32.0f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture3);
            glBindVertexArray(cubeVAO);
            glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, (void*)0);
            glBindVertexArray(0);

			//right half-door
			translateMatrix = glm::translate(identityMatrix, glm::vec3(-1.5, -2.0, 0.0));
			scaleMatrix = glm::scale(identityMatrix, glm::vec3(-1.5, 1.5, 0.125));
            rotateYMatrix = glm::rotate(identityMatrix, glm::radians(-doorAngle), glm::vec3(0.0f, 1.0f, 0.0f));
            model = translateMatrix * rotateYMatrix * scaleMatrix;
			lightingShaderWithTexture.setInt("material.diffuse", 0);
			lightingShaderWithTexture.setInt("material.specular", 1);
			lightingShaderWithTexture.setFloat("material.shininess", 32.0f);
			lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture3);
			glBindVertexArray(cubeVAO);
			glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, (void*)0);
            glBindVertexArray(0);


           
            //moi 1
            
            translateMatrix = glm::translate(identityMatrix, glm::vec3(6.5f, -1.5f, -2.0f));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.3f, 0.3f, 0.3f));
            rotateXMatrix = glm::rotate(identityMatrix, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            //translateMatrix = glm::translate(identityMatrix, sofaTranslation);
            model = rotateXMatrix * translateMatrix * scaleMatrix;
            lightingShaderWithTexture.setInt("material.diffuse", 0);
            lightingShaderWithTexture.setInt("material.specular", 1);
            lightingShaderWithTexture.setFloat("material.shininess", 0.5f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture3);
            glBindVertexArray(VAOtt);
            glDrawElements(GL_TRIANGLES, (unsigned int)tt_indices.size(), GL_UNSIGNED_INT, (void*)0);

            

           //moi 2
            translateMatrix = glm::translate(identityMatrix, glm::vec3(-12.5f, -1.5f, -2.0f));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.3f, 0.3f, 0.3f));
            rotateXMatrix = glm::rotate(identityMatrix, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            model = rotateXMatrix * translateMatrix * scaleMatrix;
            lightingShaderWithTexture.setInt("material.diffuse", 0);
            lightingShaderWithTexture.setInt("material.specular", 1);
            lightingShaderWithTexture.setFloat("material.shininess", 0.5f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture3);
            glBindVertexArray(VAOtt);
            glDrawElements(GL_TRIANGLES, (unsigned int)tt_indices.size(), GL_UNSIGNED_INT, (void*)0);



            //moi3
            translateMatrix = glm::translate(identityMatrix, glm::vec3(6.5f, 8.5f, -2.0f));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.3f, 0.3f, 0.3f));
            //angleInRadians = glm::radians(-90.0f);
            rotateXMatrix = glm::rotate(identityMatrix, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            //translateMatrix = glm::translate(identityMatrix, sofaTranslation);
            model = rotateXMatrix * translateMatrix * scaleMatrix;
            lightingShaderWithTexture.setInt("material.diffuse", 0);
            lightingShaderWithTexture.setInt("material.specular", 1);
            lightingShaderWithTexture.setFloat("material.shininess", 0.5f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture3);
            glBindVertexArray(VAOtt);
            glDrawElements(GL_TRIANGLES, (unsigned int)tt_indices.size(), GL_UNSIGNED_INT, (void*)0);

            //moi 4

            translateMatrix = glm::translate(identityMatrix, glm::vec3(-12.5f, 8.5f, -2.0f));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.3f, 0.3f, 0.3f));
            
            rotateXMatrix = glm::rotate(identityMatrix, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            //translateMatrix = glm::translate(identityMatrix, sofaTranslation);
            model = rotateXMatrix * translateMatrix * scaleMatrix;
            lightingShaderWithTexture.setInt("material.diffuse", 0);
            lightingShaderWithTexture.setInt("material.specular", 1);
            lightingShaderWithTexture.setFloat("material.shininess", 0.5f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture3);
            glBindVertexArray(VAOtt);
            glDrawElements(GL_TRIANGLES, (unsigned int)tt_indices.size(), GL_UNSIGNED_INT, (void*)0);



            //pot1

            translateMatrix = glm::translate(identityMatrix, glm::vec3(-5.9, -2.0, 0.5));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.012f, 0.015f, 0.015f));
            rotateXMatrix = glm::rotate(identityMatrix, glm::radians(-0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            //translateMatrix = glm::translate(identityMatrix, sofaTranslation);
            model = rotateXMatrix * translateMatrix * scaleMatrix;
            lightingShaderWithTexture.setInt("material.diffuse", 0);
            lightingShaderWithTexture.setInt("material.specular", 1);
            lightingShaderWithTexture.setFloat("material.shininess", 0.5f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture4);
            glBindVertexArray(VAOpot);
            glDrawElements(GL_TRIANGLES, (unsigned int)potindices.size(), GL_UNSIGNED_INT, (void*)0);


            //pot2
            translateMatrix = glm::translate(identityMatrix, glm::vec3(-0.5, -2.0, 1.0));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.012f, 0.015f, 0.015f));
            rotateXMatrix = glm::rotate(identityMatrix, glm::radians(-0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            //translateMatrix = glm::translate(identityMatrix, sofaTranslation);
            model = rotateXMatrix * translateMatrix * scaleMatrix;
            lightingShaderWithTexture.setInt("material.diffuse", 0);
            lightingShaderWithTexture.setInt("material.specular", 1);
            lightingShaderWithTexture.setFloat("material.shininess", 0.5f);
            lightingShaderWithTexture.setMat4("model", model);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture4);
            glBindVertexArray(VAOpot);
            glDrawElements(GL_TRIANGLES, (unsigned int)potindices.size(), GL_UNSIGNED_INT, (void*)0);


            //lightingshader
            lightingShader.use();
            lightingShader.setVec3("viewPos", camera.Position);
            lightingShader.setMat4("projection", projection);
            lightingShader.setMat4("view", view);


            // point light 1
            pointlight1.setUpPointLight(lightingShader);
            // point light 2
            pointlight2.setUpPointLight(lightingShader);
            // point light 3
            pointlight3.setUpPointLight(lightingShader);

            spotlight1.setUpspotLight(lightingShader);
			spotlight2.setUpspotLight(lightingShader);

            //constantShader.setMat4("view", view);
            //lightingShader.setMat4("view", view);

            lightingShader.setVec3("directionalLight.directiaon", 0.0f, -3.0f, 0.0f);
            lightingShader.setVec3("directionalLight.ambient", .5f, .5f, .5f);
            lightingShader.setVec3("directionalLight.diffuse", .8f, .8f, .8f);
            lightingShader.setVec3("directionalLight.specular", 1.0f, 1.0f, 1.0f);

            lightingShader.setBool("directionLightOn", true);


           

           
			//tree infront of dome 4
            
            // Use the tree shader program
            lightingShader.use();



            // Set model matrix
            translateMatrix = glm::translate(identityMatrix, glm::vec3(-5.5, -2.0, 1.0));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(1.8, 1.3, 1.8));
            model = translateMatrix * scaleMatrix;
            lightingShader.setMat4("model", model);

            // Set material properties
            lightingShader.setVec3("material.ambient", 0.4f, 0.2f, 0.1f);
            lightingShader.setVec3("material.diffuse", 0.4f, 0.2f, 0.1f);
            lightingShader.setVec3("material.specular", 0.5f, 0.5f, 0.5f);
            lightingShader.setFloat("material.shininess", 32.0f);

            // Draw the tree
            glBindVertexArray(treeVAO);
            glDrawArrays(GL_LINES, 0, treeVertices.size() / 6);
            glBindVertexArray(0);



            

            
            //tree infront of dome 1
            
            translateMatrix = glm::translate(identityMatrix, glm::vec3(0.0, -2.0, 1.0));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(1.8, 1.3, 1.8));
            model = translateMatrix * scaleMatrix;
            lightingShader.setMat4("model", model);

            // Set material properties
            lightingShader.setVec3("material.ambient", 0.4f, 0.2f, 0.1f);
            lightingShader.setVec3("material.diffuse", 0.4f, 0.2f, 0.1f);
            lightingShader.setVec3("material.specular", 0.5f, 0.5f, 0.5f);
            lightingShader.setFloat("material.shininess", 32.0f);

            // Draw the tree
            glBindVertexArray(treeVAO);
            glDrawArrays(GL_LINES, 0, treeVertices.size() / 6);
            glBindVertexArray(0);





            

           // moi 1
            
            // Use shader program
            lightingShader.use();
            translateMatrix = glm::translate(identityMatrix, glm::vec3(6.5f,-1.5f, -2.0f));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.3f, 0.3f, 0.3f));
            float angleInRadians = glm::radians(-90.0f);
            rotateXMatrix = glm::rotate(identityMatrix, angleInRadians, glm::vec3(1.0f, 0.0f, 0.0f));
            //translateMatrix = glm::translate(identityMatrix, sofaTranslation);
            model = rotateXMatrix * translateMatrix * scaleMatrix;
            // Define the wood color
            glm::vec4 woodColor = glm::vec4(0.52f, 0.37f, 0.26f, 1.0f); // Medium brown (wood-like)

            // Setting up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(woodColor) * 0.5f); // Darker for ambient
            lightingShader.setVec3("material.diffuse", glm::vec3(woodColor));         // Main color for diffuse
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Subtle highlights for specular
            lightingShader.setFloat("material.shininess", 16.0f);

            lightingShader.setMat4("model", model);
            // Bind VAO
            glBindVertexArray(VAOt);

            // Draw the object
           // glDrawElements(GL_TRIANGLES, tindices.size(), GL_UNSIGNED_INT, 0);


            //moi 2---------------------------------------

             // Use shader program
            lightingShader.use();
            translateMatrix = glm::translate(identityMatrix, glm::vec3(-12.5f, -1.5f, -2.0f));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.3f, 0.3f, 0.3f));
            angleInRadians = glm::radians(-90.0f);
            rotateXMatrix = glm::rotate(identityMatrix, angleInRadians, glm::vec3(1.0f, 0.0f, 0.0f));
            //translateMatrix = glm::translate(identityMatrix, sofaTranslation);
            model = rotateXMatrix * translateMatrix * scaleMatrix;
            // Define the wood color
            woodColor = glm::vec4(0.52f, 0.37f, 0.26f, 1.0f); // Medium brown (wood-like)

            // Setting up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(woodColor) * 0.5f); // Darker for ambient
            lightingShader.setVec3("material.diffuse", glm::vec3(woodColor));         // Main color for diffuse
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Subtle highlights for specular
            lightingShader.setFloat("material.shininess", 16.0f);

            lightingShader.setMat4("model", model);

            // Bind VAO
            glBindVertexArray(VAOt);

            // Draw the object
            //glDrawElements(GL_TRIANGLES, tindices.size(), GL_UNSIGNED_INT, 0);




            //moi 3---------------------------------------

              // Use shader program
            lightingShader.use();
            translateMatrix = glm::translate(identityMatrix, glm::vec3(6.5f, 8.5f, -2.0f));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.3f, 0.3f, 0.3f));
            angleInRadians = glm::radians(-90.0f);
            rotateXMatrix = glm::rotate(identityMatrix, angleInRadians, glm::vec3(1.0f, 0.0f, 0.0f));
            //translateMatrix = glm::translate(identityMatrix, sofaTranslation);
            model = rotateXMatrix * translateMatrix * scaleMatrix;
            // Define the wood color
            woodColor = glm::vec4(0.52f, 0.37f, 0.26f, 1.0f); // Medium brown (wood-like)

            // Setting up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(woodColor) * 0.5f); // Darker for ambient
            lightingShader.setVec3("material.diffuse", glm::vec3(woodColor));         // Main color for diffuse
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Subtle highlights for specular
            lightingShader.setFloat("material.shininess", 16.0f);

            lightingShader.setMat4("model", model);

            // Bind VAO
            glBindVertexArray(VAOt);

            // Draw the object
            //glDrawElements(GL_TRIANGLES, tindices.size(), GL_UNSIGNED_INT, 0);


            //moi 4---------------------------------------

             // Use shader program
            lightingShader.use();
            translateMatrix = glm::translate(identityMatrix, glm::vec3(-12.5f, 8.5f, -2.0f));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.3f, 0.3f, 0.3f));
            angleInRadians = glm::radians(-90.0f);
            rotateXMatrix = glm::rotate(identityMatrix, angleInRadians, glm::vec3(1.0f, 0.0f, 0.0f));
            //translateMatrix = glm::translate(identityMatrix, sofaTranslation);
            model = rotateXMatrix * translateMatrix * scaleMatrix;
            // Define the wood color
            woodColor = glm::vec4(0.52f, 0.37f, 0.26f, 1.0f); // Medium brown (wood-like)

            // Setting up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(woodColor) * 0.5f); // Darker for ambient
            lightingShader.setVec3("material.diffuse", glm::vec3(woodColor));         // Main color for diffuse
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Subtle highlights for specular
            lightingShader.setFloat("material.shininess", 16.0f);

            lightingShader.setMat4("model", model);

            // Bind VAO
           // glBindVertexArray(VAOt);

            // Draw the object
            glDrawElements(GL_TRIANGLES, tindices.size(), GL_UNSIGNED_INT, 0);


           




            glm::mat4 RotateTranslateMatrix = (1.0f);
            glm::mat4 InvRotateTranslateMatrix = (1.0f);

            
            // Common translation matrix for all three objects
            

            // Use shader program
            lightingShader.use();

            // First object
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.018f, 0.018f, 0.018f));
            angleInRadians = glm::radians(leftBaseRotationAngle+27.0f);
            rotateXMatrix = glm::rotate(identityMatrix, angleInRadians, glm::vec3(1.0f, 0.0f, 0.0f));
            translateMatrix = glm::translate(identityMatrix, glm::vec3(-6.5, -0.4, 2.5));
            RotateTranslateMatrix = glm::translate(identityMatrix, glm::vec3(-6.45, -2.0, 3.05));
            InvRotateTranslateMatrix = glm::translate(identityMatrix, glm::vec3(6.45, 2.0, -3.05));

            rotateYMatrix = glm::rotate(identityMatrix, angleInRadians, glm::vec3(0.0f, 1.0f, 0.0f)); // Rotate around the y-axis
            model = RotateTranslateMatrix * rotateYMatrix * InvRotateTranslateMatrix * translateMatrix * scaleMatrix;
            // Define the wood color
            glm::vec4 ironColor = glm::vec4(0.56f, 0.57f, 0.58f, 1.0f); // Metallic gray

            // Setting up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(ironColor) * 0.5f); // Darker for ambient
            lightingShader.setVec3("material.diffuse", glm::vec3(ironColor));         // Main color for diffuse
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Subtle highlights for specular
            lightingShader.setFloat("material.shininess", 16.0f);

            lightingShader.setMat4("model", model);

            // Bind VAO
            glBindVertexArray(VAOrb);

            // Draw the object
            glDrawElements(GL_TRIANGLES, rbindices.size(), GL_UNSIGNED_INT, 0);

            //------------------------------------------------------------------------------------

            // Second object
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.018f, 0.018f, 0.018f));
            angleInRadians = glm::radians(0.0f);
            rotateXMatrix = glm::rotate(identityMatrix, angleInRadians, glm::vec3(1.0f, 0.0f, 0.0f));
            translateMatrix = glm::translate(identityMatrix, glm::vec3(-6.5, -0.4, 2.5));
            model =  rotateXMatrix * translateMatrix * scaleMatrix;

            // Setting up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(ironColor) * 0.5f); // Darker for ambient
            lightingShader.setVec3("material.diffuse", glm::vec3(ironColor));         // Main color for diffuse
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Subtle highlights for specular
            lightingShader.setFloat("material.shininess", 16.0f);

            lightingShader.setMat4("model", model);

            // Bind VAO
            glBindVertexArray(VAOrbb);

            // Draw the object
            glDrawElements(GL_TRIANGLES, rbbindices.size(), GL_UNSIGNED_INT, 0);

            //------------------------------------------------------------------------------------

            // Third object
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.018f, 0.018f, 0.018f));
            angleInRadians = glm::radians(leftBaseRotationAngle+ 27.0f);
            rotateXMatrix = glm::rotate(identityMatrix, angleInRadians, glm::vec3(1.0f, 0.0f, 0.0f));
            translateMatrix = glm::translate(identityMatrix, glm::vec3(-6.5, -0.4, 2.5));
            RotateTranslateMatrix = glm::translate(identityMatrix, glm::vec3(-6.5, -2.0, 3.0));
            InvRotateTranslateMatrix = glm::translate(identityMatrix, glm::vec3(6.5, 2.0, -3.0));

            rotateYMatrix = glm::rotate(identityMatrix, angleInRadians, glm::vec3(0.0f, 1.0f, 0.0f)); // Rotate around the y-axis
            model = RotateTranslateMatrix * rotateYMatrix * InvRotateTranslateMatrix * translateMatrix * scaleMatrix;

            // Define the wood color
            woodColor = glm::vec4(0.52f, 0.37f, 0.26f, 1.0f); // Medium brown (wood-like)

            // Setting up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(ironColor) * 0.5f); // Darker for ambient
            lightingShader.setVec3("material.diffuse", glm::vec3(ironColor));         // Main color for diffuse
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Subtle highlights for specular
            lightingShader.setFloat("material.shininess", 16.0f);

            lightingShader.setMat4("model", model);

            // Bind VAO
            glBindVertexArray(VAOrbg);

            // Draw the object
            glDrawElements(GL_TRIANGLES, rbgindices.size(), GL_UNSIGNED_INT, 0);


          



		


            //---------------------------second missile


            RotateTranslateMatrix = (1.0f);
            InvRotateTranslateMatrix = (1.0f);
            //gun base
            
            // Use shader program
            lightingShader.use();
            

           // First object
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.018f, 0.018f, 0.018f));
            angleInRadians = glm::radians(rightBaseRotationAngle); // Use the common rotation angle
            translateMatrix = glm::translate(identityMatrix, glm::vec3(1.0, -0.4, 2.5));
            RotateTranslateMatrix = glm::translate(identityMatrix, glm::vec3(1.05, -2.05, 3.05));
            InvRotateTranslateMatrix = glm::translate(identityMatrix, glm::vec3(-1.05, 2.05, -3.05));

            rotateYMatrix = glm::rotate(identityMatrix, angleInRadians, glm::vec3(0.0f, 1.0f, 0.0f)); // Rotate around the y-axis
            model = RotateTranslateMatrix* rotateYMatrix * InvRotateTranslateMatrix * translateMatrix * scaleMatrix;
            


            // Define the wood color
            ironColor = glm::vec4(0.56f, 0.57f, 0.58f, 1.0f); // Metallic gray

            // Setting up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(ironColor) * 0.5f); // Darker for ambient
            lightingShader.setVec3("material.diffuse", glm::vec3(ironColor));         // Main color for diffuse
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Subtle highlights for specular
            lightingShader.setFloat("material.shininess", 16.0f);

            lightingShader.setMat4("model", model);

            // Bind VAO
            glBindVertexArray(VAOrb);

            // Draw the object
            glDrawElements(GL_TRIANGLES, rbindices.size(), GL_UNSIGNED_INT, 0);

            // Calculate new axes after rotation of the first object
            glm::vec3 xAxis(1.0f, 0.0f, 0.0f); // Local X-axis
            glm::vec3 yAxis(0.0f, 1.0f, 0.0f); // Local Y-axis

            // Transform the axes using the rotation matrix
            glm::vec3 newXAxis = glm::vec3(rotateYMatrix * glm::vec4(xAxis, 0.0f)); // Transformed X-axis
            glm::vec3 newYAxis = glm::vec3(rotateYMatrix * glm::vec4(yAxis, 0.0f)); // Transformed Y-axis



            //------------------------------------------------------------------------------------

            // Second object
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.018f, 0.018f, 0.018f));
            angleInRadians = glm::radians(0.0f);
            rotateXMatrix = glm::rotate(identityMatrix, angleInRadians, glm::vec3(1.0f, 0.0f, 0.0f));
            translateMatrix = glm::translate(identityMatrix, glm::vec3(1.0, -0.4, 2.5));
            model = rotateXMatrix * translateMatrix * scaleMatrix;

            // Setting up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(ironColor) * 0.5f); // Darker for ambient
            lightingShader.setVec3("material.diffuse", glm::vec3(ironColor));         // Main color for diffuse
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Subtle highlights for specular
            lightingShader.setFloat("material.shininess", 16.0f);

            lightingShader.setMat4("model", model);

            // Bind VAO
            glBindVertexArray(VAOrbb);

            // Draw the object
            glDrawElements(GL_TRIANGLES, rbbindices.size(), GL_UNSIGNED_INT, 0);

            //------------------------------------------------------------------------------------

            // right Gun
             
            // Right gun transformation
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.018f, 0.018f, 0.018f));
            angleInRadians = glm::radians(rightBaseRotationAngle); // Rotation angle for the gun
            translateMatrix = glm::translate(identityMatrix, glm::vec3(1.0, -0.4, 2.5));
            RotateTranslateMatrix = glm::translate(identityMatrix, glm::vec3(1, -0.75, 3.0));
            InvRotateTranslateMatrix = glm::translate(identityMatrix, glm::vec3(-1, 0.75, -3.0));

           
            rotateYMatrix = glm::rotate(identityMatrix, angleInRadians, glm::vec3(0.0f, 1.0f, 0.0f)); // Rotate around the y-axis
            model = RotateTranslateMatrix * rotateYMatrix * InvRotateTranslateMatrix * translateMatrix * scaleMatrix;






            // Define the wood color
            woodColor = glm::vec4(0.52f, 0.37f, 0.26f, 1.0f); // Medium brown (wood-like)

            // Setting up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(ironColor) * 0.5f); // Darker for ambient
            lightingShader.setVec3("material.diffuse", glm::vec3(ironColor));         // Main color for diffuse
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Subtle highlights for specular
            lightingShader.setFloat("material.shininess", 16.0f);

            lightingShader.setMat4("model", model);

            // Bind VAO
            glBindVertexArray(VAOrbg);

            // Draw the object
            glDrawElements(GL_TRIANGLES, rbgindices.size(), GL_UNSIGNED_INT, 0);

            glBindVertexArray(0); // Optional for safety

            //-------------------------------------solid tree1

            glm::vec3 parentTransform = glm::vec3(-8.0, 0.0, -4.0);

            translateMatrix = glm::translate(identityMatrix, parentTransform + glm::vec3(-4.5, 0.0, 0.5));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.0004f, 0.00030f, 0.00028f));
            model = translateMatrix *  scaleMatrix;

            // Setting up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(0.13f, 0.55f, 0.13f) * 0.5f); // Darker for ambient
            lightingShader.setVec3("material.diffuse", glm::vec3(0.13f, 0.55f, 0.13f));         // Main color for diffuse
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Subtle highlights for specular
            lightingShader.setFloat("material.shininess", 16.0f);

            lightingShader.setMat4("model", model);

            // Bind VAO
           glBindVertexArray(VAOtr);

            // Draw the object
           glDrawElements(GL_TRIANGLES, trindices.size(), GL_UNSIGNED_INT, 0);


             
            glBindVertexArray(0); // Optional for safety

           

            //-------------------------------------trunk
            translateMatrix = glm::translate(identityMatrix, parentTransform+glm::vec3(-4.0, -1.5, 0.5));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.00072f, 0.00072f, 0.00072f));
            model = translateMatrix * scaleMatrix;

            // Setting up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(woodColor) * 0.5f); // Darker for ambient
            lightingShader.setVec3("material.diffuse", glm::vec3(woodColor));         // Main color for diffuse
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Subtle highlights for specular
            lightingShader.setFloat("material.shininess", 16.0f);

            lightingShader.setMat4("model", model);

            // Bind VAO
            glBindVertexArray(VAOtrunk);

            // Draw the object
            glDrawElements(GL_TRIANGLES, trunkindices.size(), GL_UNSIGNED_INT, 0);



            glBindVertexArray(0); // Optional for safety

            //-------------------------------------solid tree2

            glm::vec3 parentTransform1 = glm::vec3(8.0, 0.0, -4.0);

            translateMatrix = glm::translate(identityMatrix, parentTransform1 + glm::vec3(-5.0, 0.0, 0.5));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.0004f, 0.00030f, 0.00028f));
            model = translateMatrix * scaleMatrix;

            // Setting up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(0.13f, 0.55f, 0.13f) * 0.5f); // Darker for ambient
            lightingShader.setVec3("material.diffuse", glm::vec3(0.13f, 0.55f, 0.13f));         // Main color for diffuse
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Subtle highlights for specular
            lightingShader.setFloat("material.shininess", 16.0f);

            lightingShader.setMat4("model", model);

            // Bind VAO
            glBindVertexArray(VAOtr);

            // Draw the object
            glDrawElements(GL_TRIANGLES, trindices.size(), GL_UNSIGNED_INT, 0);



            glBindVertexArray(0); // Optional for safety



            //-------------------------------------trunk2
            translateMatrix = glm::translate(identityMatrix, parentTransform1 + glm::vec3 (-4.0f, -1.5f, 0.5f));
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.00072f, 0.00072f, 0.00072f));
            //
            model =  translateMatrix * scaleMatrix;

            // Setting up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(woodColor)); // Darker for ambient
            lightingShader.setVec3("material.diffuse", glm::vec3(woodColor));         // Main color for diffuse
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Subtle highlights for specular
            lightingShader.setFloat("material.shininess", 16.0f);

            lightingShader.setMat4("model", model);

            // Bind VAO
            glBindVertexArray(VAOtrunk);

            // Draw the object
            glDrawElements(GL_TRIANGLES, trunkindices.size(), GL_UNSIGNED_INT, 0);



            glBindVertexArray(0); // Optional for safety

			//-------------------------------------missile 1
            // Initialize position for the object
            //objectPosition = glm::vec3(-4.0f, -2.0f, 0.5f); // Initial position

            
            // Update position along the diagonal of the yz-plane when the P key is pressed
            if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
               
				//update the object position
				objectPosition2.y += 0.01f;
				objectPosition1.y += 0.01f;
				objectPosition3.y += 0.01f;
				objectPosition4.y += 0.01f;


               



            }

            //missile 1

               // Compute the model matrix
            glm::mat4 translatemMatrix = glm::translate(identityMatrix, objectPosition1);
            glm::mat4 scalemMatrix = glm::scale(identityMatrix, glm::vec3(0.0125f, 0.0125f, 0.0125f));
            angleInRadians = glm::radians(45.0); // Use the common rotation angle
            float angle = glm::radians(leftBaseRotationAngle);
            RotateTranslateMatrix = glm::translate(identityMatrix, glm::vec3(-6.5, -0.4, 2.5));
            glm::mat4 pivotTranslateMatrix = glm::translate(identityMatrix, glm::vec3(-6.4, -1.8, 3.05));
            InvRotateTranslateMatrix = glm::translate(identityMatrix, glm::vec3(6.5, 0.4, -2.5));

            rotateXMatrix = glm::rotate(identityMatrix, angleInRadians, glm::vec3(1.0f, 0.0f, 0.0f)); // Rotate around the y-axis
            glm::mat4 rotatetest = glm::rotate(identityMatrix, angle, glm::vec3(0.0f, 1.0f, 0.0f)); // Rotate around the y-axis

            //model = RotateTranslateMatrix * rotateXMatrix * InvRotateTranslateMatrix * translateMatrix * scaleMatrix;

            glm::mat4 mmodel = RotateTranslateMatrix * rotatetest * InvRotateTranslateMatrix * RotateTranslateMatrix * rotateXMatrix * InvRotateTranslateMatrix * translatemMatrix * scalemMatrix;


            // Set up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(ironColor)); // Ambient light
            lightingShader.setVec3("material.diffuse", glm::vec3(ironColor)); // Diffuse light
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Specular highlights
            lightingShader.setFloat("material.shininess", 16.0f);

            // Pass the model matrix to the shader
            lightingShader.setMat4("model", mmodel);

            // Bind VAO
            glBindVertexArray(VAOm);

            // Draw the object
            glDrawElements(GL_TRIANGLES, missileindices.size(), GL_UNSIGNED_INT, 0);

            // Unbind VAO (optional for safety)
            glBindVertexArray(0);

            //-------------------------------------missile 2
            // Compute the model matrix
            translatemMatrix = glm::translate(identityMatrix, objectPosition2);
            scalemMatrix = glm::scale(identityMatrix, glm::vec3(0.0125f, 0.0125f, 0.0125f));
            angleInRadians = glm::radians(45.0); // Use the common rotation angle
            angle = glm::radians(leftBaseRotationAngle);
            RotateTranslateMatrix = glm::translate(identityMatrix, glm::vec3(-6.5, -0.4, 2.5));
            //pivotTranslateMatrix = glm::translate(identityMatrix, glm::vec3(-6.4, -1.8, 3.05));
            InvRotateTranslateMatrix = glm::translate(identityMatrix, glm::vec3(6.5, 0.4, -2.5));

            rotateXMatrix = glm::rotate(identityMatrix, angleInRadians, glm::vec3(1.0f, 0.0f, 0.0f)); // Rotate around the y-axis
            rotatetest = glm::rotate(identityMatrix, angle, glm::vec3(0.0f, 1.0f, 0.0f)); // Rotate around the y-axis

            //model = RotateTranslateMatrix * rotateXMatrix * InvRotateTranslateMatrix * translateMatrix * scaleMatrix;

            mmodel = RotateTranslateMatrix * rotatetest * InvRotateTranslateMatrix * RotateTranslateMatrix * rotateXMatrix * InvRotateTranslateMatrix * translatemMatrix * scalemMatrix;


            // Set up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(ironColor)); // Ambient light
            lightingShader.setVec3("material.diffuse", glm::vec3(ironColor)); // Diffuse light
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Specular highlights
            lightingShader.setFloat("material.shininess", 16.0f);

            // Pass the model matrix to the shader
            lightingShader.setMat4("model", mmodel);

            // Bind VAO
            glBindVertexArray(VAOm);

            // Draw the object
            glDrawElements(GL_TRIANGLES, missileindices.size(), GL_UNSIGNED_INT, 0);

            // Unbind VAO (optional for safety)
            glBindVertexArray(0);



            //-------------------------------------missile 3
            // Compute the model matrix
            translatemMatrix = glm::translate(identityMatrix, objectPosition3);
            scalemMatrix = glm::scale(identityMatrix, glm::vec3(0.0125f, 0.0125f, 0.0125f));
            angleInRadians = glm::radians(45.0); // Use the common rotation angle
            angle = glm::radians(leftBaseRotationAngle);
            RotateTranslateMatrix = glm::translate(identityMatrix, glm::vec3(-6.5, -0.4, 2.5));
            //pivotTranslateMatrix = glm::translate(identityMatrix, glm::vec3(-6.4, -1.8, 3.05));
            InvRotateTranslateMatrix = glm::translate(identityMatrix, glm::vec3(6.5, 0.4, -2.5));

            rotateXMatrix = glm::rotate(identityMatrix, angleInRadians, glm::vec3(1.0f, 0.0f, 0.0f)); // Rotate around the y-axis
            rotatetest = glm::rotate(identityMatrix, angle, glm::vec3(0.0f, 1.0f, 0.0f)); // Rotate around the y-axis

            //model = RotateTranslateMatrix * rotateXMatrix * InvRotateTranslateMatrix * translateMatrix * scaleMatrix;

            mmodel = RotateTranslateMatrix * rotatetest * InvRotateTranslateMatrix * RotateTranslateMatrix * rotateXMatrix * InvRotateTranslateMatrix * translatemMatrix * scalemMatrix;


            // Set up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(ironColor)); // Ambient light
            lightingShader.setVec3("material.diffuse", glm::vec3(ironColor)); // Diffuse light
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Specular highlights
            lightingShader.setFloat("material.shininess", 16.0f);

            // Pass the model matrix to the shader
            lightingShader.setMat4("model", mmodel);

            // Bind VAO
            glBindVertexArray(VAOm);

            // Draw the object
            glDrawElements(GL_TRIANGLES, missileindices.size(), GL_UNSIGNED_INT, 0);

            // Unbind VAO (optional for safety)
            glBindVertexArray(0);


            //-------------------------------------missile 4
            // Compute the model matrix
            translatemMatrix = glm::translate(identityMatrix, objectPosition4);
            scalemMatrix = glm::scale(identityMatrix, glm::vec3(0.0125f, 0.0125f, 0.0125f));
            angleInRadians = glm::radians(45.0); // Use the common rotation angle
            angle = glm::radians(leftBaseRotationAngle);
            RotateTranslateMatrix = glm::translate(identityMatrix, glm::vec3(-6.5, -0.4, 2.5));
            pivotTranslateMatrix = glm::translate(identityMatrix, glm::vec3(-6.45, -1.8, 3.05));
            InvRotateTranslateMatrix = glm::translate(identityMatrix, glm::vec3(6.5, 0.4, -2.5));

            rotateXMatrix = glm::rotate(identityMatrix, angleInRadians, glm::vec3(1.0f, 0.0f, 0.0f)); // Rotate around the y-axis
            rotatetest = glm::rotate(identityMatrix, angle, glm::vec3(0.0f, 1.0f, 0.0f)); // Rotate around the y-axis

            //model = RotateTranslateMatrix * rotateXMatrix * InvRotateTranslateMatrix * translateMatrix * scaleMatrix;

            mmodel = RotateTranslateMatrix * rotatetest * InvRotateTranslateMatrix * RotateTranslateMatrix * rotateXMatrix * InvRotateTranslateMatrix * translatemMatrix * scalemMatrix;


            // Set up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(ironColor)); // Ambient light
            lightingShader.setVec3("material.diffuse", glm::vec3(ironColor)); // Diffuse light
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Specular highlights
            lightingShader.setFloat("material.shininess", 16.0f);

            // Pass the model matrix to the shader
            lightingShader.setMat4("model", mmodel);

            // Bind VAO
            glBindVertexArray(VAOm);

            // Draw the object
            glDrawElements(GL_TRIANGLES, missileindices.size(), GL_UNSIGNED_INT, 0);

            // Unbind VAO (optional for safety)
            glBindVertexArray(0);


            // Update the parameter t
                // Update position along the diagonal of the yz-plane when the P key is pressed
            if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) {
                objectPosition2.y += translationSpeed * deltaTime; // Move along the Y axis
                objectPosition1.y += translationSpeed * deltaTime; // Move along the Z axis
                objectPosition3.y += translationSpeed * deltaTime;
                objectPosition4.y += translationSpeed * deltaTime;
            }




            // Compute the model matrix
            translatemMatrix = glm::translate(identityMatrix, glm::vec3(-13.0, -2.0, 6.5));
            scalemMatrix = glm::scale(identityMatrix, glm::vec3(0.025f, 0.025f, 0.025f));

            mmodel = translatemMatrix * scalemMatrix;


            // Set up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(ironColor)); // Ambient light
            lightingShader.setVec3("material.diffuse", glm::vec3(ironColor)); // Diffuse light
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Specular highlights
            lightingShader.setFloat("material.shininess", 16.0f);

            // Pass the model matrix to the shader
            lightingShader.setMat4("model", mmodel);

            // Bind VAO
            glBindVertexArray(VAOm);

            // Draw the object
            glDrawElements(GL_TRIANGLES, missileindices.size(), GL_UNSIGNED_INT, 0);

            // Unbind VAO (optional for safety)
            glBindVertexArray(0);


            // Compute the model matrix
            translatemMatrix = glm::translate(identityMatrix, glm::vec3(6.5, -2.0, 6.0));
            scalemMatrix = glm::scale(identityMatrix, glm::vec3(0.025f, 0.025f, 0.025f));

            mmodel = translatemMatrix * scalemMatrix;


            // Set up material properties
            lightingShader.setVec3("material.ambient", glm::vec3(ironColor)); // Ambient light
            lightingShader.setVec3("material.diffuse", glm::vec3(ironColor)); // Diffuse light
            lightingShader.setVec3("material.specular", glm::vec3(0.2f, 0.2f, 0.2f)); // Specular highlights
            lightingShader.setFloat("material.shininess", 16.0f);

            // Pass the model matrix to the shader
            lightingShader.setMat4("model", mmodel);

            // Bind VAO
            glBindVertexArray(VAOm);

            // Draw the object
            glDrawElements(GL_TRIANGLES, missileindices.size(), GL_UNSIGNED_INT, 0);

            // Unbind VAO (optional for safety)
            glBindVertexArray(0);


            
           

           


     
        if (openDoor) {
            if (doorAngle < 90.0f) {
                doorAngle += 0.25;
            }
        }

        if (!openDoor) {
            if (doorAngle > 0.0f) {
                doorAngle -= 0.25;
            }
        }



        lightingShader.use();
        lightingShader.setVec3("viewPos", camera.Position);
        lightingShader.setMat4("projection", projection);
        lightingShader.setMat4("view", view);


        // point light 1
        pointlight1.setUpPointLight(lightingShader);
        // point light 2
        pointlight2.setUpPointLight(lightingShader);
        // point light 3
        pointlight3.setUpPointLight(lightingShader);

        spotlight1.setUpspotLight(lightingShader);
        
        //constantShader.setMat4("view", view);
        //lightingShader.setMat4("view", view);

        for (unsigned int i = 0; i < 3; i++)
        {
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.5f, 0.5f, 0.5f));
            translateMatrix = glm::translate(identityMatrix, glm::vec3(0.12, -0.1, 0) + pointLightPositions[i]); // Adjust the offset as needed
            model = translateMatrix * scaleMatrix;
            drawSphere(VAO_S, lightingShader, glm::vec3(1.0f, 1.0f, 0.8f), model, indices_s);



        }

        
        
        ourShader.use();
        ourShader.setMat4("projection", projection);

        //glm::mat4 view = basic_camera.createViewMatrix();
        ourShader.setMat4("view", view);
        
        glBindVertexArray(lightCubeVAO);
        for (unsigned int i = 0; i < 3; i++)
        {
            model = glm::mat4(1.0f);
            model = glm::translate(model, glm::vec3(0.05, -3.5, 0)+pointLightPositions[i]);
            model = glm::scale(model, glm::vec3(0.1f, 3.0f, 0.1f)); // Make it a smaller cube
            ourShader.setMat4("model", model);
            ourShader.setVec4("color", glm::vec4(0.3f, 0.3f, 0.3f, 1.0f));
            glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
            //glDrawArrays(GL_TRIANGLES, 0, 36);
         


        }
       
        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &cubeVAO);
    glDeleteVertexArrays(1, &lightCubeVAO);
    glDeleteBuffers(1, &cubeVAO);
    glDeleteBuffers(1, &cubeVAO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------



void lightEffect(unsigned int VAO, Shader lightShader, glm::mat4 model, glm::vec3 color)
{
    lightShader.use();
    lightShader.setVec3("material.ambient", color);
    lightShader.setVec3("material.diffuse", color);
    lightShader.setVec3("material.specular", glm::vec3(0.5f, 0.5f, 0.5f));
    lightShader.setFloat("material.shininess", 32.0f);

    lightShader.setMat4("model", model);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, (void*)0);
    glBindVertexArray(0);
}

void drawCube(unsigned int VAO, Shader shader, glm::mat4 model, glm::vec4 color)
{
    shader.setMat4("model", model);
    shader.setVec4("color", color);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
}






long long nCr(int n, int r)
{
    if (r > n / 2)
        r = n - r; // because C(n, r) == C(n, n - r)
    long long ans = 1;
    int i;

    for (i = 1; i <= r; i++)
    {
        ans *= n - r + i;
        ans /= i;
    }

    return ans;
}

void BezierCurve(double t, float xy[2], GLfloat ctrlpoints[], int L)
{
    double y = 0;
    double x = 0;
    t = t > 1.0 ? 1.0 : t;
    for (int i = 0; i < L + 1; i++)
    {
        long long ncr = nCr(L, i);
        double oneMinusTpow = pow(1 - t, double(L - i));
        double tPow = pow(t, double(i));
        double coef = oneMinusTpow * tPow * ncr;
        x += coef * ctrlpoints[i * 3];
        y += coef * ctrlpoints[(i * 3) + 1];

    }
    xy[0] = float(x);
    xy[1] = float(y);
}

unsigned int hollowBezier(GLfloat ctrlpoints[], int L, vector<float>& coordinates, vector<float>& normals, vector<int>& indices, vector<float>& vertices, float div = 1.0)
{
    int i, j;
    float x, y, z, r;                //current coordinates
    float theta;
    float nx, ny, nz, lengthInv;    // vertex normal


    const float dtheta =   (2*pi / ntheta)/div;        //angular step size

    float t = 0;
    float dt = 1.0 / nt;
    float xy[2];
    vector <float> textureCoords;
    for (i = 0; i <= nt; ++i)              //step through y
    {
        BezierCurve(t, xy, ctrlpoints, L);
        r = xy[0];
        y = xy[1];
        theta = 0;
        t += dt;
        lengthInv = 1.0 / r;

        for (j = 0; j <= ntheta; ++j)
        {
            double cosa = cos(theta);
            double sina = sin(theta);
            z = r * cosa;
            x = r * sina;

            coordinates.push_back(x);
            coordinates.push_back(y);
            coordinates.push_back(z);

            // normalized vertex normal (nx, ny, nz)
            // center point of the circle (0,y,0)
            nx = (x - 0) * lengthInv;
            ny = (y - y) * lengthInv;
            nz = (z - 0) * lengthInv;

            normals.push_back(nx);
            normals.push_back(ny);
            normals.push_back(nz);

            // Texture coordinates (u, v)
            float u = float(j) / float(ntheta); // Around the circle
            float v = float(i) / float(nt);     // Along the curve
            textureCoords.push_back(u);
            textureCoords.push_back(v);

            theta += dtheta;
        }
    }

    // generate index list of triangles
    // k1--k1+1
    // |  / |
    // | /  |
    // k2--k2+1

    int k1, k2;
    for (int i = 0; i < nt; ++i)
    {
        k1 = i * (ntheta + 1);     // beginning of current stack
        k2 = k1 + ntheta + 1;      // beginning of next stack

        for (int j = 0; j < ntheta; ++j, ++k1, ++k2)
        {
            // k1 => k2 => k1+1
            indices.push_back(k1);
            indices.push_back(k2);
            indices.push_back(k1 + 1);

            // k1+1 => k2 => k2+1
            indices.push_back(k1 + 1);
            indices.push_back(k2);
            indices.push_back(k2 + 1);
        }
    }

    size_t count = coordinates.size();
    for (int i = 0; i < count; i += 3)
    {
        vertices.push_back(coordinates[i]);
        vertices.push_back(coordinates[i + 1]);
        vertices.push_back(coordinates[i + 2]);

        vertices.push_back(normals[i]);
        vertices.push_back(normals[i + 1]);
        vertices.push_back(normals[i + 2]);

        // Add texture coordinates
        vertices.push_back(textureCoords[i / 3 * 2]);
        vertices.push_back(textureCoords[i / 3 * 2 + 1]);
    }

    unsigned int bezierVAO;
    glGenVertexArrays(1, &bezierVAO);
    glBindVertexArray(bezierVAO);

    // create VBO to copy vertex data to VBO
    unsigned int bezierVBO;
    glGenBuffers(1, &bezierVBO);
    glBindBuffer(GL_ARRAY_BUFFER, bezierVBO);           // for vertex data
    glBufferData(GL_ARRAY_BUFFER,                   // target
        (unsigned int)vertices.size() * sizeof(float), // data size, # of bytes
        vertices.data(),   // ptr to vertex data
        GL_STATIC_DRAW);                   // usage

    // create EBO to copy index data
    unsigned int bezierEBO;
    glGenBuffers(1, &bezierEBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bezierEBO);   // for index data
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,           // target
        (unsigned int)indices.size() * sizeof(unsigned int),             // data size, # of bytes
        indices.data(),               // ptr to index data
        GL_STATIC_DRAW);                   // usage

    // activate attrib arrays
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    // set attrib arrays with stride and offset
    int stride = 32;     // should be 32 bytes
    glVertexAttribPointer(0, 3, GL_FLOAT, false, stride, (void*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, false, stride, (void*)(sizeof(float) * 3));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float))); // Texture Coord
    

    // unbind VAO, VBO and EBO
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    return bezierVAO;
}

/*unsigned int hollowBezier(GLfloat ctrlpoints[], int L, vector<float>& coordinates, vector<float>& normals, vector<int>& indices, vector<float>& vertices, float div = 1.0)
{
    int i, j;
    float x, y, z, r;                //current coordinates
    float theta;
    float nx, ny, nz;                // vertex normal

    const float PI = 3.14159265358979323846f;
    const int ntheta = 36; // Assuming ntheta is defined somewhere
    const int nt = 36; // Assuming nt is defined somewhere
    const float dtheta = (2 * PI / ntheta) / div; // angular step size

    float t = 0;
    float dt = 1.0f / nt;
    float xy[2];
    vector<float> textureCoords;
    for (i = 0; i <= nt; ++i) // step through y
    {
        BezierCurve(t, xy, ctrlpoints, L);
        r = xy[0];
        y = xy[1];
        theta = 0;
        t += dt;

        for (j = 0; j <= ntheta; ++j)
        {
            float cosa = cos(theta);
            float sina = sin(theta);
            z = r * cosa;
            x = r * sina;

            coordinates.push_back(x);
            coordinates.push_back(y);
            coordinates.push_back(z);

            // Calculate tangent vectors
            glm::vec3 tangent1(-r * sina, 0, r * cosa);
            glm::vec3 tangent2(0, 1, 0);

            // Calculate normal using cross product
            glm::vec3 normal = glm::normalize(glm::cross(tangent1, tangent2));

            normals.push_back(normal.x);
            normals.push_back(normal.y);
            normals.push_back(normal.z);

            // Texture coordinates (u, v)
            float u = float(j) / float(ntheta); // Around the circle
            float v = float(i) / float(nt);     // Along the curve
            textureCoords.push_back(u);
            textureCoords.push_back(v);

            theta += dtheta;
        }
    }

    // generate index list of triangles
    // k1--k1+1
    // |  / |
    // | /  |
    // k2--k2+1

    int k1, k2;
    for (int i = 0; i < nt; ++i)
    {
        k1 = i * (ntheta + 1);     // beginning of current stack
        k2 = k1 + ntheta + 1;      // beginning of next stack

        for (int j = 0; j < ntheta; ++j, ++k1, ++k2)
        {
            // k1 => k2 => k1+1
            indices.push_back(k1);
            indices.push_back(k2);
            indices.push_back(k1 + 1);

            // k1+1 => k2 => k2+1
            indices.push_back(k1 + 1);
            indices.push_back(k2);
            indices.push_back(k2 + 1);
        }
    }

    size_t count = coordinates.size();
    for (int i = 0; i < count; i += 3)
    {
        vertices.push_back(coordinates[i]);
        vertices.push_back(coordinates[i + 1]);
        vertices.push_back(coordinates[i + 2]);

        vertices.push_back(normals[i]);
        vertices.push_back(normals[i + 1]);
        vertices.push_back(normals[i + 2]);

        // Add texture coordinates
        vertices.push_back(textureCoords[i / 3 * 2]);
        vertices.push_back(textureCoords[i / 3 * 2 + 1]);
    }

    unsigned int bezierVAO;
    glGenVertexArrays(1, &bezierVAO);
    glBindVertexArray(bezierVAO);

    // create VBO to copy vertex data to VBO
    unsigned int bezierVBO;
    glGenBuffers(1, &bezierVBO);
    glBindBuffer(GL_ARRAY_BUFFER, bezierVBO);           // for vertex data
    glBufferData(GL_ARRAY_BUFFER,                   // target
        (unsigned int)vertices.size() * sizeof(float), // data size, # of bytes
        vertices.data(),   // ptr to vertex data
        GL_STATIC_DRAW);                   // usage

    // create EBO to copy index data
    unsigned int bezierEBO;
    glGenBuffers(1, &bezierEBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bezierEBO);   // for index data
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,           // target
        (unsigned int)indices.size() * sizeof(unsigned int),             // data size, # of bytes
        indices.data(),               // ptr to index data
        GL_STATIC_DRAW);                   // usage

    // activate attrib arrays
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    // set attrib arrays with stride and offset
    int stride = 32;     // should be 32 bytes
    glVertexAttribPointer(0, 3, GL_FLOAT, false, stride, (void*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, false, stride, (void*)(sizeof(float) * 3));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float))); // Texture Coord


    // unbind VAO, VBO and EBO
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    return bezierVAO;
}*/




void read_file(string file_name, vector<float>& vec)
{
    ifstream file(file_name);
    float number;

    while (file >> number)
        vec.push_back(number);

    file.close();
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    /*
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) translate_Y += 0.01;
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) translate_Y -= 0.01;
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) translate_X += 0.01;
    if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS) translate_X -= 0.01;
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) translate_Z += 0.01;
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) translate_Z -= 0.01;
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) scale_X += 0.01;
    if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS) scale_X -= 0.01;
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS) scale_Y += 0.01;
    if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS) scale_Y -= 0.01;
    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS) scale_Z += 0.01;
    if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS) scale_Z -= 0.01;
    */

    /*if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
    {
        rotateAngle_X += 1;
    }
    if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS)
    {
        rotateAngle_Y += 1;
    }
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
    {
        rotateAngle_Z += 1;
    }

    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS)
    {
        eyeX += 2.5 * deltaTime;
        basic_camera.eye = glm::vec3(eyeX, eyeY, eyeZ);
    }
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
    {
        eyeX -= 2.5 * deltaTime;
        basic_camera.eye = glm::vec3(eyeX, eyeY, eyeZ);

        //cout << "x: "<<eyeX << endl;
    }
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS)
    {
        eyeZ += 2.5 * deltaTime;
        basic_camera.eye = glm::vec3(eyeX, eyeY, eyeZ);
    }
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
    {
        eyeZ -= 2.5 * deltaTime;
        basic_camera.eye = glm::vec3(eyeX, eyeY, eyeZ);
        //cout << "z: " << eyeZ << endl;
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    {
        eyeY += 2.5 * deltaTime;
        basic_camera.eye = glm::vec3(eyeX, eyeY, eyeZ);
        //cout << "y: " << eyeY << endl;
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
    {
        eyeY -= 2.5 * deltaTime;
        basic_camera.eye = glm::vec3(eyeX, eyeY, eyeZ);
    }
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
    {
        lookAtX += 2.5 * deltaTime;
        basic_camera.lookAt = glm::vec3(lookAtX, lookAtY, lookAtZ);
    }
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
    {
        lookAtX -= 2.5 * deltaTime;
        basic_camera.lookAt = glm::vec3(lookAtX, lookAtY, lookAtZ);
    }
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS)
    {
        lookAtY += 2.5 * deltaTime;
        basic_camera.lookAt = glm::vec3(lookAtX, lookAtY, lookAtZ);
    }
    if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS)
    {
        lookAtY -= 2.5 * deltaTime;
        basic_camera.lookAt = glm::vec3(lookAtX, lookAtY, lookAtZ);
    }
    */


    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        camera.ProcessKeyboard(FORWARD, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        camera.ProcessKeyboard(LEFT, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        camera.ProcessKeyboard(RIGHT, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        camera.ProcessKeyboard(UP, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        camera.ProcessKeyboard(DOWN, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) {
        camera.ProcessKeyboard(P_UP, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS) {
        camera.ProcessKeyboard(P_DOWN, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) {
        camera.ProcessKeyboard(Y_LEFT, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
        camera.ProcessKeyboard(Y_RIGHT, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
        camera.ProcessKeyboard(R_LEFT, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS) {
        camera.ProcessKeyboard(R_RIGHT, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) on = true;
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) on = false;
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS) birdEye = true;
    if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS) birdEye = false;

    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) {
        leftBaseRotationAngle += 1.0f; // Increase the rotation angle
        if (leftBaseRotationAngle >= 360.0f) {
            leftBaseRotationAngle -= 360.0f; // Keep the angle within 0-360 degrees
        }
    }

    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) {
        rightBaseRotationAngle += 1.0f; // Increase the common rotation angle
        if (rightBaseRotationAngle >= 360.0f) {
            rightBaseRotationAngle -= 360.0f; // Keep the angle within 0-360 degrees
        }
    }

    /*// Update position along the diagonal of the yz-plane when the P key is pressed
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
        objectPosition.y += translationSpeed * deltaTime; // Move along the Y axis
        objectPosition.z += translationSpeed * deltaTime; // Move along the Z axis
    }*/

    if (birdEye) {
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            cameraPos.z -= birdEyeSpeed * deltaTime;
            target.z -= birdEyeSpeed * deltaTime;
            if (cameraPos.z <= 4.0) {
                cameraPos.z = 4.0;
            }

            if (target.z <= -3.5) {
                target.z = -3.5;
            }
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            cameraPos.z += birdEyeSpeed * deltaTime;
            target.z += birdEyeSpeed * deltaTime;
            /*cout << "tgt: " << target.z << endl;
            cout << "pos: " << cameraPos.z << endl;*/
            if (cameraPos.z >= 13.5) {
                cameraPos.z = 13.5;
            }
            if (target.z >= 6.0) {
                target.z = 6.0;
            }
        }

        /*if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
        {
            dl = false;

        }
        if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
        {
            dl = true;

        }*/




    }


}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{


    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
    {
        if (dl)
            dl = false;
        else
            dl = true;

    }
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
    {
        if (point1)
        {
            point1 = false;
            pointlight1.turnOff();
        }
        else
        {
            point1 = true;
            pointlight1.turnOn();
        }

    }
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS)
    {
        if (point2)
        {
            point2 = false;
            pointlight2.turnOff();
        }
        else
        {
            point2 = true;
            pointlight2.turnOn();
        }

    }
    if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS)
    {
		if (spt)
		{
			spt = false;
			spotlight1.turnOff();
			spotlight2.turnOff();
		}
		else
		{
			spt = true;
			spotlight1.turnOn();
			spotlight2.turnOn();
		}

    }
    if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS)
    {
        if (point3)
        {
            point3 = false;
            pointlight3.turnOff();
        }
        else
        {
            point3 = true;
            pointlight3.turnOn();
        }

    }

    if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS)
    {
        if (ambientToggle)
        {
            pointlight1.turnAmbientOff();
            pointlight2.turnAmbientOff();
            ambientToggle = false;
        }
        else
        {
            pointlight1.turnAmbientOn();
            pointlight2.turnAmbientOn();
            ambientToggle = true;
        }

    }

    if (glfwGetKey(window, GLFW_KEY_7) == GLFW_PRESS)
    {
        if (diffuseToggle)
        {
            pointlight1.turnDiffuseOff();
            pointlight2.turnDiffuseOff();
            //d_def_on = 0.0f;

            diffuseToggle = false;
        }
        else
        {
            pointlight1.turnDiffuseOn();
            pointlight2.turnDiffuseOn();

            //d_def_on = 1.0f;
            diffuseToggle = true;
        }

    }

    if (glfwGetKey(window, GLFW_KEY_8) == GLFW_PRESS)
    {
        if (specularToggle)
        {
            pointlight1.turnSpecularOff();
            pointlight2.turnSpecularOff();
            //d_def_on = 0.0f;

            specularToggle = false;
        }
        else
        {
            pointlight1.turnSpecularOn();
            pointlight2.turnSpecularOn();

            //d_def_on = 1.0f;
            specularToggle = true;
        }

        

    }

   
    

    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS) {
        openDoor = true;
    }

    if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS) {
        openDoor = false;
    }



}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}


// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}



// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

unsigned int loadTexture(char const* path, GLenum textureWrappingModeS, GLenum textureWrappingModeT, GLenum textureFilteringModeMin, GLenum textureFilteringModeMax)
{
    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data)
    {
        GLenum format;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, textureWrappingModeS);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, textureWrappingModeT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, textureFilteringModeMin);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, textureFilteringModeMax);

        stbi_image_free(data);
    }
    else
    {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}



void load_texture(unsigned int& texture, string image_name, GLenum format)
{
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(image_name.c_str(), &width, &height, &nrChannels, 0);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        cout << "Failed to load texture " << image_name << endl;
    }
    stbi_image_free(data);
}
