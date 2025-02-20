#ifndef MEC_SERVER_H
#define MEC_SERVER_H

#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/socket.h"

namespace ns3 {

class MecServer : public Application
{
public:
  static TypeId GetTypeId(void);
  MecServer();
  virtual ~MecServer();
  void HandleRead(Ptr<Socket> socket);

protected:
  virtual void StartApplication(); // Called at start
  virtual void StopApplication();  // Called at stop

 private:
    Ptr<Socket> m_socket;
    Ipv6Address m_broadcastAddress;
    uint16_t m_port;
    void SendModelUpdate();

};

} // namespace ns3

#endif // MEC_SERVER_H
