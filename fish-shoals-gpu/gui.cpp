#include "gui.h"

Gui::Gui(Fishes* fishes)
{
    memcpy(&simulationParams, &fishes->params, sizeof(simulationParams));
    coh = &fishes->params.cohesion;
    sep = &fishes->params.separation;
    alg = &fishes->params.alignment;
    max_speed = &fishes->params.max_speed;
    min_speed = &fishes->params.min_speed;
    margin = &fishes->params.margin;
    turn = &fishes->params.turn;
    visibility = &fishes->params.visibility;
}

void Gui::update()
{

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Control Panel", &show_window);

    ImGui::SeparatorText("Press M to unlock free camera");
    ImGui::SeparatorText("Press N to release mouse pointer");

    if (ImGui::CollapsingHeader("Simulation parameteres"))
    {
        ImGui::SliderFloat("Separation", sep,
            simulationParams.MIN_SEPARATION, simulationParams.MAX_SEPARATION);
        ImGui::SliderFloat("Alignment", alg,
            simulationParams.MIN_ALIGNMENT, simulationParams.MAX_ALIGNMENT);
        ImGui::SliderFloat("Cohesion", coh,
            simulationParams.MIN_COHESION, simulationParams.MAX_COHESION);
        ImGui::SliderFloat("Border margin", margin,
            simulationParams.MIN_MARGIN, simulationParams.MAX_MARGIN);
        ImGui::SliderFloat("Max speed", max_speed,
            *min_speed, simulationParams.MAX_MAX_SPEED);
        ImGui::SliderFloat("Min speed", min_speed,
            simulationParams.MIN_MIN_SPEED, *max_speed);
        ImGui::SliderFloat("Visibility", visibility,
            simulationParams.MIN_VISIBILITY, simulationParams.MAX_VISIBILITY);
        ImGui::SliderFloat("Turn", turn,
            simulationParams.MIN_TURN, simulationParams.MAX_TURN);
    }

    if (ImGui::CollapsingHeader("Simulation information"))
    {
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
