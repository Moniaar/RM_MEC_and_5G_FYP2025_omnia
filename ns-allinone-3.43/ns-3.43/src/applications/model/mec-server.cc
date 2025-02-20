#include "ns3/mec-server.h"
#include "ns3/socket-factory.h"
#include "ns3/simulator.h"
#include "ns3/socket.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("MecServer");
NS_OBJECT_ENSURE_REGISTERED(MecServer);

TypeId MecServer::GetTypeId(void) {
  static TypeId tid = TypeId("ns3::MecServer")
    .SetParent<Application>()
    .SetGroupName("Applications")
    .AddConstructor<MecServer>();
  return tid;
}

MecServer::MecServer() {}

MecServer::~MecServer() {}

void MecServer::StartApplication() {
    if (!m_socket) {
        m_socket = Socket::CreateSocket(GetNode(), TypeId::LookupByName("ns3::UdpSocketFactory"));
        Inet6SocketAddress local = Inet6SocketAddress(Ipv6Address::GetAny(), m_port);
        m_socket->Bind(local);
    }

    SendModelUpdate(); // Send initial model update
}

void MecServer::SendModelUpdate() {
    std::string modelWeights = "UpdatedModelWeights"; // Simulated model update
    Ptr<Packet> packet = Create<Packet>((uint8_t*)modelWeights.c_str(), modelWeights.length());

    Inet6SocketAddress remote = Inet6SocketAddress(m_broadcastAddress, m_port);
    m_socket->SendTo(packet, 0, remote);

    std::cout << "Sent model update to IoT devices." << std::endl;
}


void MecServer::StopApplication() {
  NS_LOG_INFO("MEC Server Stopped");

  if (m_socket) {
    m_socket->Close();
    m_socket = 0;
  }
}

// Fix: Add HandleRead implementation
void MecServer::HandleRead(Ptr<Socket> socket) {
  NS_LOG_INFO("MEC Server received data");
}

} // namespace ns3
