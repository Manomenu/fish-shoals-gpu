#include "gui.h"

Gui::Gui(Fishes* fishes)
{
    memcpy(&simulationParams, &fishes->params, sizeof(simulationParams));
}

void Gui::update(float fps)
{
    this->fps = fps;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Control Panel", &show_window);

    ImGui::SeparatorText("Press CTRL to release mouse pointer");

    if (ImGui::CollapsingHeader("Simulation parameteres"))
    {
        ImGui::SliderFloat("Separation", &simulationParams.separation,
            simulationParams.MIN_SEPARATION, simulationParams.MAX_SEPARATION);
        ImGui::SliderFloat("Alignment", &simulationParams.alignment,
            simulationParams.MIN_ALIGNMENT, simulationParams.MAX_ALIGNMENT);
        ImGui::SliderFloat("Cohesion", &simulationParams.cohesion,
            simulationParams.MIN_COHESION, simulationParams.MAX_COHESION);
        ImGui::SliderFloat("Border margin", &simulationParams.margin,
            simulationParams.MIN_MARGIN, simulationParams.MAX_MARGIN);
        ImGui::SliderFloat("Speed", &simulationParams.speed,
            simulationParams.MIN_SPEED, simulationParams.MAX_SPEED);
        ImGui::SliderFloat("Visibility", &simulationParams.visibility,
            simulationParams.MIN_VISIBILITY, simulationParams.MAX_VISIBILITY);
    }

    if (ImGui::CollapsingHeader("Simulation information"))
    {
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "%d fps", fps);
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "%d fishes", FISH_COUNT);
    }

    if (ImGui::CollapsingHeader("Control information"))
    {
        ImGui::BulletText("W - move camera forward");
        ImGui::BulletText("A - move camera left");
        ImGui::BulletText("S - move camera right");
        ImGui::BulletText("D - move camera back");
    }


    ImGui::End();
}
