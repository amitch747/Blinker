// Receives relative dx dy via TCP and injects relative mouse movement with smoothing

#include <winsock2.h>
#pragma comment(lib,"ws2_32.lib") // winsock is not default or something, need to explicitly link

#include <windows.h>
#include <ws2tcpip.h>
#include <string>
#include <iostream>
#include <deque>


const int SMOOTHING_FRAMES = 10;

class MouseSmoother {
private:
    std::deque<int> dxBuffer;
    std::deque<int> dyBuffer;

public:
    void add(int dx, int dy) {
        if (dxBuffer.size() >= SMOOTHING_FRAMES) dxBuffer.pop_front();
        if (dyBuffer.size() >= SMOOTHING_FRAMES) dyBuffer.pop_front();
        dxBuffer.push_back(dx);
        dyBuffer.push_back(dy);
    }

    POINT getSmoothedDelta() {
        int sumX = 0, sumY = 0;
        for (int d : dxBuffer) sumX += d;
        for (int d : dyBuffer) sumY += d;
        int n = static_cast<int>(dxBuffer.size());
        
        if (n == 0) 
            return { 0, 0 };

        return { sumX / n, sumY / n }; // Returns an initialized POINT
    }
};

void MoveMouseRelative(int dx, int dy)
{
    if (dx == 0 && dy == 0) 
        return;

    INPUT in = {};
    in.type = INPUT_MOUSE;
    in.mi.dx = dx;
    in.mi.dy = dy;
    in.mi.dwFlags = MOUSEEVENTF_MOVE ;
    SendInput(1, &in, sizeof(in));
}

int main()
{
    WSADATA wsa;  
    WSAStartup(MAKEWORD(2, 2), &wsa);

    SOCKET server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    sockaddr_in addr{};  
    addr.sin_family = AF_INET;  // ipv4
    addr.sin_port = htons(747);
    addr.sin_addr.s_addr = INADDR_ANY;
    bind(server, (sockaddr*)&addr, sizeof(addr));  

    listen(server, SOMAXCONN);
    std::cout << "cursor_server waiting for blinker - port 54000\n";
    std::cout << std::flush;  // Force flush

    SOCKET client = accept(server, nullptr, nullptr);
    std::cout << "blinker connected\n";
    std::cout << std::flush;  // Force flush

    int flag = 1;
    setsockopt(client, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(flag)); // Avoid grouping

    char buf[128];
    std::string bigBuf;
    MouseSmoother smoother;

    while (true)
    {
        int n = recv(client, buf, sizeof(buf), 0);
        if (n <= 0) 
            break;

        bigBuf.append(buf, n);

        size_t newlineIndex;
        while ((newlineIndex = bigBuf.find('\n')) != std::string::npos)
        {
            std::string cursorMessage = bigBuf.substr(0, newlineIndex);
            bigBuf.erase(0, newlineIndex + 1);

            int dx, dy;
            if (sscanf_s(cursorMessage.c_str(), "%d %d", &dx, &dy) == 2)
            {
                smoother.add(dx, dy);
                POINT smoothed = smoother.getSmoothedDelta();
                MoveMouseRelative(smoothed.x, smoothed.y);
            }
        }

    }

    closesocket(client);  closesocket(server);  WSACleanup();
    std::cout << "cursor_server shutdown\n";
    std::cout << std::flush;  // Force flush

    return 0;
}