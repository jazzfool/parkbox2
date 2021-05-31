#include "gfx/context.hpp"
#include "gfx/renderer.hpp"

#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>
#include <spdlog/pattern_formatter.h>

struct LevelFormatter : public spdlog::custom_flag_formatter {
    void format(const spdlog::details::log_msg& msg, const std::tm&, spdlog::memory_buf_t& dest) override {
        std::string s;
        switch (msg.level) {
        case spdlog::level::level_enum::info: {
            s = "INFO";
            break;
        }
        case spdlog::level::level_enum::warn: {
            s = "WARN";
            break;
        }
        case spdlog::level::level_enum::err: {
            s = "FAIL";
            break;
        }
        default:
            break;
        }

        dest.append(&s[0], &s[s.length()]);
    }

    std::unique_ptr<spdlog::custom_flag_formatter> clone() const override {
        return std::make_unique<LevelFormatter>();
    }
};

int main() {
    std::unique_ptr<spdlog::pattern_formatter> formatter = std::make_unique<spdlog::pattern_formatter>();
    formatter->add_flag<LevelFormatter>('y').set_pattern("%^[%y]%$ %v");
    spdlog::set_formatter(std::move(formatter));

    gfx::vk_log(volkInitialize());
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Parkbox", nullptr, nullptr);

    gfx::Context cx;
    cx.init(window);

    gfx::Renderer renderer;
    cx.renderer = &renderer;
    renderer.init(cx);

    renderer.run();

    gfx::vk_log(vkDeviceWaitIdle(cx.dev));

    renderer.cleanup();
    cx.cleanup();
}
