package sd.server;

public class Main {

    public static void main(String[] args) {
        System.out.println("Server starting...");
        Server server = new Server(10, 30);
        server.run(8080);
    }
}

