# NS-3
### How everything works?:
1. Write your server code in this path ```src/applications/model/``` . It should be in C++, and place the header of the same file in the same path.
2. Afterwards go back 1 directory in the applications directory place the files you just wrote below the other sources in the CMakeLists.txt file. Both the C++ and the header.
3. Go backwards to write a simulation for what you just added in the scratch folder.
4. Run your simulation using ./ns3 run scratch/file_name.cc
5. Run Netanim by copying the simulation file output in XML format and bring it to the netanim directory behind aka: netanim-3.109. Then run Netanim using ```./Netanim``` command.
6. Open and play with the parameters in the Netanim.
### If netanim doesn't work we use the following steps:
1. cat /etc/resolv.conf : to check your IPV4 address and instead of this I personally use ```ipconfig``` in my powershell to know my IPV4 address.
2. write export DISPLAY= then your IPV4 Address and :0.0 afterwards: here is an example:```export DISPLAY=10.255.255.254:0.0```
3. export LIBGL_ALWAYS_INDIRECT=1
4. sudo apt install x11-apps
5. Run ./NetAnim  :).

## What if Netanim doesn't start?
we have multiple options to follow:
It looks like **NetAnim** is failing to connect to the **X server** due to missing authentication or incorrect display settings. Follow these steps to fix it:

---

### **1. Verify Your X Server Connection**
Since you're using `export DISPLAY=172.26.192.1:0.0`, ensure that:
- **The IP address is correct** (`172.26.192.1` should be your host machine's IP).
- **Your X server is running** on your host machine (if using WSL, you need an X server like **VcXsrv** or **X410**).

To check if your X server is running, try:

```sh
xclock
```

If `xclock` does not start, the issue is with your X server.

---

### **2. Allow X Server Connections**
Run the following command **on your host machine** (not inside WSL or the remote machine):

```sh
xhost +local:*
```

If you're using **WSL**, try:

```sh
export DISPLAY=:0
```

Then, re-run NetAnim:

```sh
./NetAnim
```

---

### **3. Check Qt Platform Plugin**
The error mentions:

```
Could not load the Qt platform plugin "xcb"
```

Ensure that the required Qt dependencies are installed:

```sh
sudo apt install qt5-default qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
```

If already installed, explicitly specify the Qt plugin when running NetAnim:

```sh
./NetAnim -platform xcb
```

---

### **4. Run NetAnim Without Display Variables**
If you're on **a local Linux machine**, try resetting the `DISPLAY` variable:

```sh
unset DISPLAY
./NetAnim
```

If you're using **WSL2**, use:

```sh
export DISPLAY=:0
./NetAnim
```

---

### **5. Run in Offscreen Mode (If Needed)**
If the issue persists, try running NetAnim in **offscreen mode**:

```sh
./NetAnim -platform offscreen
```

This will let you generate animation traces without displaying the UI.

---
### Which one of these is the MEC server node?:
instead of this in scratch : ```s(interfaces.GetAddress(iotNodes.GetN() - 1), port));```
 I put:
```
clientSocket->Connect(InetSocketAddress(interfaces.GetAddress(numIoTDevices), port));
```
This ensures the IoT devices send their specs to the MEC server (the last node in mecServerNode), not to another IoT device.

Answer: Which Node is the Server?
The MEC server is the last node created, which is mecServerNode.Get(0).
Since iotNodes.Create(numIoTDevices); creates 4 IoT nodes (indices 0 to 3), and mecServerNode.Create(1); creates one more node (index 4), the MEC server should be node 4.

## Why **the MEC (Multi-access Edge Computing) server is implemented at the application layer** while **IoT devices are handled in the network layer**?:
This is is due to the functional roles of each component in the system.  

### **1️⃣ MEC Server at the Application Layer**  
- The **MEC server provides services**, such as **processing, computation, and storage** for IoT devices.  
- It acts as an **edge-cloud system**, meaning it runs application-layer logic (e.g., handling requests, aggregating data, distributing tasks).  
- Since it is handling **higher-level tasks** (not just forwarding packets but making decisions on the received data), it fits within the **application layer**.  
- It typically uses **sockets** to communicate with IoT devices, just like a web server handling client requests.  

✅ **Example**:  
- A **MEC server** can receive **sensor data from IoT devices**, process it using AI models, and then send back a response.  

---

### **2️⃣ IoT Devices at the Network Layer**  
- IoT devices mainly handle **data transmission, connectivity, and routing**, which are **network layer** concerns.  
- They typically interact with the **MEC server via network protocols** (e.g., TCP, UDP, IPv4, IPv6).  
- They **don’t process or store data significantly**; instead, they **send data** to the MEC server.  
- IoT devices require **mobility models** and **networking modules**, so their behavior is mainly defined within the network layer.  

✅ **Example**:  
- A **temperature sensor** in an IoT device **collects data**, packages it into a **network packet**, and **transmits it** to the MEC server.  

---

## What to do if Netanim is misbehaving?
This error suggests an issue with Qt, which NetAnim relies on. Here are a few things you can try to fix it:

### 1. **Ensure Required Qt Packages Are Installed**
Run:
```bash
sudo apt update
sudo apt install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
```
Then, recompile NetAnim:
```bash
cd ~/ns-allinone-3.43/netanim-3.109
qmake NetAnim.pro
make clean
make
```
Try running `./NetAnim` again.

---

### 2. **Run With Software Rendering**
Some Qt applications crash due to OpenGL issues. Try launching NetAnim with:
```bash
export QT_XCB_GL_INTEGRATION=none
./NetAnim
```

---

### 3. **Check for Missing Dependencies**
Use:
```bash
ldd NetAnim | grep "not found"
```
If any dependencies are missing, install them.

---

### 4. **Run With Debugger**
To get more details on the error:
```bash
gdb ./NetAnim
run
```
If it crashes, type `bt` (backtrace) to analyze the issue.

---


### RF or Pission and why?
