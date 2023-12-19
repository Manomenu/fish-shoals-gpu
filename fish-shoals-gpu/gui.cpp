#include "gui.h"

void Gui::update()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Control Panel", &show_window);

    // Create an scene mode section
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Sample text");

    ImGui::End();
}
