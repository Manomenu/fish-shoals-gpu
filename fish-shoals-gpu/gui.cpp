#include "gui.h"

Gui::Gui(Fishes* fishes)
{
    memcpy(&simulationParams, &fishes->params, sizeof(simulationParams));
    coh = &fishes->params.cohesion;
    sep = &fishes->params.separation;
    alg = &fishes->params.alignment;
    coh_alter = &fishes->params.cohesion_alter;
    sep_alter = &fishes->params.separation_alter;
    alg_alter = &fishes->params.alignment_alter;
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

    if (ImGui::CollapsingHeader("Simulation parameteres"))
    {
        ImGui::SeparatorText("Shoal 1 parameters");
        ImGui::SliderFloat("Separation 1", sep,
            simulationParams.MIN_SEPARATION, simulationParams.MAX_SEPARATION);
        ImGui::SliderFloat("Alignment 1", alg,
            simulationParams.MIN_ALIGNMENT, simulationParams.MAX_ALIGNMENT);
        ImGui::SliderFloat("Cohesion 1", coh,
            simulationParams.MIN_COHESION, simulationParams.MAX_COHESION);
        ImGui::SeparatorText("Shoal 2 parameters");
        ImGui::SliderFloat("Separation 2", sep_alter,
            simulationParams.MIN_SEPARATION, simulationParams.MAX_SEPARATION);
        ImGui::SliderFloat("Alignment 2", alg_alter,
            simulationParams.MIN_ALIGNMENT, simulationParams.MAX_ALIGNMENT);
        ImGui::SliderFloat("Cohesion 2", coh_alter,
            simulationParams.MIN_COHESION, simulationParams.MAX_COHESION);
        ImGui::SeparatorText("Other");
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
        #ifdef CPU
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "Simulation mode: CPU");
        #else
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "Simulation mode: GPU");
        #endif
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "Fishes: %d", FISH_COUNT);
    }

    if (ImGui::CollapsingHeader("Control information"))
    {
        ImGui::BulletText("ESC - close program");
        ImGui::BulletText("W   - move camera forward");
        ImGui::BulletText("A   - move camera left");
        ImGui::BulletText("S   - move camera right");
        ImGui::BulletText("D   - move camera back");
        ImGui::BulletText("M   - unlock free camera");
        ImGui::BulletText("N   - release mouse pointer");
        ImGui::BulletText("K   - pause simulation");
        ImGui::BulletText("L   - unpause simulation");
    }


    ImGui::End();
}
