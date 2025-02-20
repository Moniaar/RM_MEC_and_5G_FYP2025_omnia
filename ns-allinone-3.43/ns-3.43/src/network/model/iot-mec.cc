#include "iot-device.h"
#include <random>
#include "ns3/socket.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("IoTDevice");

// Constructor
IoTDevice::IoTDevice(double cpu, double energy, double bandwidth, bool charge)
    : m_cpuFrequency(cpu), m_energy(energy), m_bandwidth(bandwidth), m_wirelessCharging(charge) {}

// Getters
double IoTDevice::GetCpuFrequency() const { return m_cpuFrequency; }
double IoTDevice::GetEnergy() const { return m_energy; }
double IoTDevice::GetBandwidth() const { return m_bandwidth; }
bool IoTDevice::HasWirelessCharging() const { return m_wirelessCharging; }

// Function to generate multiple IoT devices
std::vector<IoTDevice> GenerateIoTDevices(int numDevices) {
    std::vector<IoTDevice> devices;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> cpuDist(0.1, 1.0);
    std::uniform_real_distribution<double> bandwidthDist(0.1, 2.0);

    for (int i = 0; i < numDevices; i++) {
        double cpu = cpuDist(generator);
        double bandwidth = bandwidthDist(generator);
        double energy = 5000 + (cpu * 1000);
        devices.emplace_back(cpu, energy, bandwidth, true);
    }
    return devices;
}

// IoT Device receives model updates
void IoTDevice::StartApplication() {
    if (!m_socket) {
        m_socket = Socket::CreateSocket(GetNode(), TypeId::LookupByName("ns3::UdpSocketFactory"));
        Inet6SocketAddress local = Inet6SocketAddress(Ipv6Address::GetAny(), m_port);
        m_socket->Bind(local);
        m_socket->SetRecvCallback(MakeCallback(&IoTDevice::ReceiveModelUpdate, this));
    }
}

void IoTDevice::ReceiveModelUpdate(Ptr<Socket> socket) {
    Ptr<Packet> packet;
    Address from;
    while ((packet = socket->RecvFrom(from))) {
        uint8_t buffer[1024];
        packet->CopyData(buffer, packet->GetSize());
        std::string receivedData(reinterpret_cast<char*>(buffer), packet->GetSize());

        NS_LOG_INFO("IoT Device Received Model Update: " << receivedData);
    }
}

} // namespace ns3
