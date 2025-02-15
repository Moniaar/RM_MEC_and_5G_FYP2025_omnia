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
