#include "ns3/mec-server.h"
#include "ns3/socket-factory.h"
#include "ns3/simulator.h"
#include "ns3/socket.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("MecServer");
NS_OBJECT_ENSURE_REGISTERED(MecServer);

// static method that returns a TypeId for the MecServer class.
// Declares a static TypeId object (tid) associated with "ns3::MecServer", meaning the MecServer is uniquely identified as "ns3::MecServer" in ns-3.
TypeId MecServer::GetTypeId(void) {
  static TypeId tid = TypeId("ns3::MecServer")
    .SetParent<Application>()
    .SetGroupName("Applications")
    // Objects can be created dynamically via CreateObject<MecServer>().
    .AddConstructor<MecServer>();
  return tid;
}

MecServer::MecServer() {}

MecServer::~MecServer() {}

void MecServer::StartApplication() {
    if (!m_socket) {
        m_socket = Socket::CreateSocket(GetNode(), TypeId::LookupByName("ns3::UdpSocketFactory"));
        Inet6SocketAddress local = Inet6SocketAddress(Ipv6Address::GetAny(), m_port);
        // Binds the socket to an address and port meaning it will listen for any incoming connections.
        m_socket->Bind(local);
    }

    SendModelUpdate(); // To send an initial message to IoT devices.
}

// Creates a simulated model update message (modelWeights).
// Converts the message into an ns-3 packet.
// Sends the packet to a broadcast address (m_broadcastAddress) over UDP.
// Displays a message confirming the update was sent.
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

// Handles incoming messages from IoT devices.
// Right now, it only logs that data was received.
// You can modify this function to process the received data.
void MecServer::HandleRead(Ptr<Socket> socket) {
  NS_LOG_INFO("MEC Server received data");
}

} // namespace ns3
