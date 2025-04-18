#include "rclcpp/rclcpp.hpp"
class MyNode : public rclcpp::Node
{
public:
    MyNode() : Node("parallel_parking_node")
    {
        RCLCPP_INFO(this->get_logger(), "CPP Node has been started");
    }
private:
};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MyNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
}