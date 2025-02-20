#ifndef IOT_DEVICE_H
#define IOT_DEVICE_H

#include "ns3/core-module.h"
#include "ns3/socket.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/ipv6-address.h"
#include "ns3/names.h"
#include "ns3/node.h"
#include "ns3/application.h"  // Ensure IoTDevice inherits from Application

namespace ns3 {

class IoTDevice : public Application {  // Inherit from Application
public:
    IoTDevice(double cpu, double energy, double bandwidth, bool charge);

    double GetCpuFrequency() const;
    double GetEnergy() const;
    double GetBandwidth() const;
    bool HasWirelessCharging() const;

    virtual void StartApplication() override;  // Correct function signature
    void ReceiveModelUpdate(Ptr<Socket> socket);

private:
    double m_cpuFrequency;
    double m_energy;
    double m_bandwidth;
    bool m_wirelessCharging;

    Ptr<Socket> m_socket;
    uint16_t m_port;  // Port number for communication
};

std::vector<IoTDevice> GenerateIoTDevices(int numDevices);

} // namespace ns3

#endif // IOT_DEVICE_H
