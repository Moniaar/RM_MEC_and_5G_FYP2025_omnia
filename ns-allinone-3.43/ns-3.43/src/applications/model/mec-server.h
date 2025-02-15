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

protected:
  virtual void StartApplication(); // Called at start
  virtual void StopApplication();  // Called at stop

 private:
  void HandleRead(Ptr<Socket> socket); // Declare missing function

  Ptr<Socket> m_socket;
  EventId m_event;
};

} // namespace ns3

#endif // MEC_SERVER_H
