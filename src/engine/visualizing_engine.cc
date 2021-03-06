//
// Created by wei on 17-10-22.
//

#include <glog/logging.h>
#include "engine/visualizing_engine.h"

const std::string kShaderPath = "../src/extern/opengl-wrapper/shader";

VisualizingEngine::VisualizingEngine(std::string window_name,
                                     int width,
                                     int height) {
  Init(window_name, width, height);
}

VisualizingEngine::~VisualizingEngine() {
  ray_caster_.Free();
  compact_mesh_.Free();
  bounding_box_.Free();
  trajectory_.Free();
}

void VisualizingEngine::Init(std::string window_name, int width, int height) {
  window_.Init(window_name, width, height);
  camera_.set_projection(width, height);
  glm::mat4 m = glm::mat4(1.0f);
  m[1][1] = -1;
  m[2][2] = -1;
  camera_.set_model(m);
}

int VisualizingEngine::Render() {
  glClearColor(1, 1, 1, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  RenderMain();
  if (enable_bounding_box_ || enable_trajectory_)
    RenderHelper();

  window_.swap_buffer();
  if (window_.get_key(GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    return -1;
  }
  return 0;
}

void VisualizingEngine::set_interaction_mode(
    bool enable_interaction
) {
  enable_interaction_ = enable_interaction;
  camera_.SwitchInteraction(enable_interaction);
}

void VisualizingEngine::update_view_matrix() {
  camera_.UpdateView(window_);
  view_ = camera_.view();
  mvp_ = camera_.mvp();
}

void VisualizingEngine::set_view_matrix(glm::mat4 view) {
  view_ = camera_.model() * view * glm::inverse(camera_.model());
  mvp_ = camera_.projection() * view_ * camera_.model();
}

void VisualizingEngine::set_light(gl::Light &light) {
  light_ = light;
}
void VisualizingEngine::LoadLight(std::string path) {
  light_.Load(path);
}

void VisualizingEngine::BindMainProgram(
    uint max_vertices,
    uint max_triangles,
    bool enable_global_mesh,
    bool enable_polygon_mode,
    bool enable_color
) {
  if (!enable_color) {
    main_program_.Load(kShaderPath + "/img_vtx_normal_uni_light_vert.glsl",
                       gl::kVertexShader);
    main_program_.ReplaceMacro("LIGHT_COUNT", light_.count_str,
                               gl::kVertexShader);
    main_program_.Load(kShaderPath + "/img_vtx_normal_uni_light_frag.glsl",
                       gl::kFragmentShader);
    main_program_.ReplaceMacro("LIGHT_COUNT", light_.count_str,
                               gl::kFragmentShader);
    main_program_.Build();

    main_uniforms_.GetLocation(main_program_.id(), "mvp", gl::kMatrix4f);
    main_uniforms_.GetLocation(main_program_.id(), "view", gl::kMatrix4f);
    main_uniforms_.GetLocation(main_program_.id(), "light", gl::kVector3f);
    main_uniforms_.GetLocation(main_program_.id(), "light_power", gl::kFloat);
    main_uniforms_.GetLocation(main_program_.id(),
                               "ambient_coeff",
                               gl::kVector3f);
    main_uniforms_.GetLocation(main_program_.id(),
                               "specular_coeff",
                               gl::kVector3f);
    main_uniforms_.GetLocation(main_program_.id(),
                               "diffuse_color",
                               gl::kVector3f);
    main_uniforms_.GetLocation(main_program_.id(),
                               "light_color",
                               gl::kVector3f);

  } else {
    main_program_.Load(kShaderPath + "/img_vtx_color_vert.glsl",
                       gl::kVertexShader);
    main_program_.Load(kShaderPath + "/img_vtx_color_frag.glsl",
                       gl::kFragmentShader);
    main_program_.Build();

    main_uniforms_.GetLocation(main_program_.id(), "mvp", gl::kMatrix4f);
  }

  main_args_.Init(3, true);
  main_args_.InitBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                        max_vertices);
  main_args_.InitBuffer(1, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                        max_vertices);
  main_args_.InitBuffer(2,
                        {GL_ELEMENT_ARRAY_BUFFER, sizeof(int), 3,
                         GL_UNSIGNED_INT},
                        max_triangles);

  enable_global_mesh_ = enable_global_mesh;
  enable_polygon_mode_ = enable_polygon_mode;
  enable_color_ = enable_color;
};

void VisualizingEngine::BindMainUniforms() {
  main_uniforms_.Bind("mvp", &mvp_, 1);

  if (!enable_color_) {
    main_uniforms_.Bind("mvp", &mvp_, 1);
    main_uniforms_.Bind("view", &view_, 1);
    main_uniforms_.Bind("light", light_.positions.data(), light_.count);
    main_uniforms_.Bind("light_power", &light_.power, 1);
    main_uniforms_.Bind("ambient_coeff", &light_.ambient_coeff, 1);
    main_uniforms_.Bind("specular_coeff", &light_.specular_coeff, 1);
    main_uniforms_.Bind("diffuse_color", &light_.diffuse_color, 1);
    main_uniforms_.Bind("light_color", &light_.light_color, 1);
  }
}

void VisualizingEngine::BindMainData() {
  main_args_.BindBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                        compact_mesh_.vertex_count(), compact_mesh_.vertices());
  if (enable_color_)
    main_args_.BindBuffer(1, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                          compact_mesh_.vertex_count(), compact_mesh_.colors());
  else
    main_args_.BindBuffer(1,
                          {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                          compact_mesh_.vertex_count(),
                          compact_mesh_.normals());
  main_args_.BindBuffer(2,
                        {GL_ELEMENT_ARRAY_BUFFER, sizeof(int), 3,
                         GL_UNSIGNED_INT},
                        compact_mesh_.triangle_count(),
                        compact_mesh_.triangles());
}

void VisualizingEngine::RenderMain() {
  glUseProgram(main_program_.id());
  BindMainUniforms();
  BindMainData();

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  /// NOTE: Use GL_UNSIGNED_INT instead of GL_INT, otherwise it won't work
  if (enable_polygon_mode_) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  }
  glDrawElements(GL_TRIANGLES,
                 compact_mesh_.triangle_count() * 3,
                 GL_UNSIGNED_INT,
                 0);
}

void VisualizingEngine::BuildHelperProgram() {
  helper_program_.Load(kShaderPath + "/img_uni_color_vert.glsl",
                       gl::kVertexShader);
  helper_program_.Load(kShaderPath + "/img_uni_color_frag.glsl",
                       gl::kFragmentShader);
  helper_program_.Build();

  helper_uniforms_.GetLocation(helper_program_.id(), "mvp", gl::kMatrix4f);
  helper_uniforms_.GetLocation(helper_program_.id(),
                               "uni_color",
                               gl::kVector3f);
}

void VisualizingEngine::BindHelperUniforms() {
  glm::vec3 color = glm::vec3(1, 0, 0);
  helper_uniforms_.Bind("mvp", &mvp_, 1);
  helper_uniforms_.Bind("uni_color", &color, 1);
}

void VisualizingEngine::InitBoundingBoxData(
    uint max_vertices
) {
  enable_bounding_box_ = true;
  bounding_box_.Resize(max_vertices);
  box_args_.Init(1, true);
  box_args_.InitBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                       max_vertices);
}

void VisualizingEngine::BindBoundingBoxData() {
  box_args_.BindBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                       bounding_box_.vertex_count(),
                       bounding_box_.vertices());
}

void VisualizingEngine::InitTrajectoryData(
    uint max_vertices
) {
  enable_trajectory_ = true;
  trajectory_.Alloc(max_vertices);
  trajectory_args_.Init(1, true);
  trajectory_args_.InitBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                              max_vertices);
}

void VisualizingEngine::BindTrajectoryData() {
  trajectory_args_.BindBuffer(0, {GL_ARRAY_BUFFER, sizeof(float), 3, GL_FLOAT},
                              trajectory_.vertex_count(),
                              trajectory_.vertices());
}

void VisualizingEngine::RenderHelper() {
  if (enable_bounding_box_) {
    glUseProgram(helper_program_.id());
    BindHelperUniforms();

    BindBoundingBoxData();
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(5.0f);
    glDrawArrays(GL_LINES, 0, bounding_box_.vertex_count());
  }

  if (enable_trajectory_) {
    glUseProgram(helper_program_.id());
    BindHelperUniforms();

    BindTrajectoryData();
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(5.0f);
    glDrawArrays(GL_LINES, 0, trajectory_.vertex_count());
  }
}

void VisualizingEngine::BuildRayCaster(
    const RayCasterParams &ray_caster_params
) {
  enable_ray_casting_ = true;
  ray_caster_.Alloc(ray_caster_params);
}

void VisualizingEngine::RenderRayCaster(
    float4x4 view,
    BlockArray &blocks,
    HashTable &hash_table,
    GeometryHelper &geometry_helper
) {
  ray_caster_.Cast(hash_table,
                   blocks,
                   0,
                   ray_caster_.data(),
                   geometry_helper,
                   view);
  cv::imshow("RayCasting", ray_caster_.surface_image());
  cv::waitKey(1);
}