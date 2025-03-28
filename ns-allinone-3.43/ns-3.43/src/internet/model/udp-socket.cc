/*
 * Copyright (c) 2007 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "udp-socket.h"

#include "ns3/boolean.h"
#include "ns3/integer.h"
#include "ns3/log.h"
#include "ns3/object.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/uinteger.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("UdpSocket");

NS_OBJECT_ENSURE_REGISTERED(UdpSocket);

TypeId
UdpSocket::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::UdpSocket")
            .SetParent<Socket>()
            .SetGroupName("Internet")
            .AddAttribute(
                "RcvBufSize",
                "UdpSocket maximum receive buffer size (bytes)",
                UintegerValue(131072),
                MakeUintegerAccessor(&UdpSocket::GetRcvBufSize, &UdpSocket::SetRcvBufSize),
                MakeUintegerChecker<uint32_t>())
            .AddAttribute("IpTtl",
                          "socket-specific TTL for unicast IP packets (if non-zero)",
                          UintegerValue(0),
                          MakeUintegerAccessor(&UdpSocket::GetIpTtl, &UdpSocket::SetIpTtl),
                          MakeUintegerChecker<uint8_t>())
            .AddAttribute(
                "IpMulticastTtl",
                "socket-specific TTL for multicast IP packets (if non-zero)",
                UintegerValue(0),
                MakeUintegerAccessor(&UdpSocket::GetIpMulticastTtl, &UdpSocket::SetIpMulticastTtl),
                MakeUintegerChecker<uint8_t>())
            .AddAttribute(
                "IpMulticastIf",
                "interface index for outgoing multicast on this socket; -1 indicates to use "
                "default interface",
                IntegerValue(-1),
                MakeIntegerAccessor(&UdpSocket::GetIpMulticastIf, &UdpSocket::SetIpMulticastIf),
                MakeIntegerChecker<int32_t>())
            .AddAttribute(
                "IpMulticastLoop",
                "whether outgoing multicast sent also to loopback interface",
                BooleanValue(false),
                MakeBooleanAccessor(&UdpSocket::GetIpMulticastLoop, &UdpSocket::SetIpMulticastLoop),
                MakeBooleanChecker())
            .AddAttribute(
                "MtuDiscover",
                "If enabled, every outgoing ip packet will have the DF flag set.",
                BooleanValue(false),
                MakeBooleanAccessor(&UdpSocket::SetMtuDiscover, &UdpSocket::GetMtuDiscover),
                MakeBooleanChecker());
    return tid;
}

UdpSocket::UdpSocket()
{
    NS_LOG_FUNCTION(this);
}

UdpSocket::~UdpSocket()
{
    NS_LOG_FUNCTION(this);
}

} // namespace ns3
