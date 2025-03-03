# DevEnvInfoServer - Cursor MCP Server for Development Environment Information

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/carterlasalle/system_information_mcp)
[![smithery badge](https://smithery.ai/badge/@carterlasalle/system_information_mcp)](https://smithery.ai/server/@carterlasalle/system_information_mcp)

This project implements a Cursor Model Context Protocol (MCP) server that provides detailed information about your development environment to the Cursor code editor.  By leveraging this server, Cursor's intelligent agent can gain a deeper understanding of your system's configuration, installed tools, and running processes, enabling more context-aware and helpful assistance.

## Features

This MCP server provides the following information categories about your development environment:

* **System Information:**
    * Operating System Version and Platform Details
    * Hardware Details (Processor, Machine, System Architecture)
    * Python Versions and Locations
    * Installed Package Managers (brew, npm, pip, yarn, uv, conda) and their versions
    * Virtual Environment Information (detected environments and active environment)
    * System Locale and Timezone
    * Top 20 Environment Variables
    * Available Shells (bash, zsh, fish, sh, powershell, cmd.exe)
    * Simplified Firewall and Network Configurations (OS-dependent)
* **Development Environment Details:**
    * Installed Compilers and Interpreters (gcc, clang, javac, node, ruby, perl, php, ghc, rustc, go)
    * Jupyter Kernels and Running Containers (Docker, Podman)
    * Virtual Machines (Hyper-V, VMware, VirtualBox)
    * GPU and CUDA Information (NVIDIA GPUs and CUDA Compiler Version)
    * Top Running Development Processes and Services
* **Python Specific Information:**
    * Installed Python Packages (pip, conda, poetry, pyenv)
    * Python Site-Packages Locations
    * Active Python Environments
* **Package Manager Details:**
    * Homebrew Installed Packages (macOS and Linux)
    * Global Packages (npm, yarn, Rust toolchain, Go environment)
* **Configuration and Dotfiles:**
    * Shell Configuration Files (.bashrc, .zshrc, .profile, .bash_profile, .config/fish/config.fish)
    * Git, NPM, and Editor Configurations (VSCode, JetBrains, Neovim)
    * Shell Aliases, Functions, and Custom Scripts (from shell config files)
* **Installed Applications:**
    * Installed IDEs and Extensions (VSCode, JetBrains, Vim, Emacs)
    * System Installed Applications (Simplified List)
* **System and Hardware Performance (Simplified Metrics):**
    * CPU Load Average
    * Battery and Power Management Configurations
    * Temperature Sensors and Fan Speeds
* **Network and Security (Simplified):**
    * Running Network Services and Open Ports
    * VPN and Proxy Settings
    * SSH Keys and Active Connections
    * Simplified Firewall Logs and Rules
* **Containerization and Virtualization:**
    * WSL (Windows Subsystem for Linux)
    * Docker and Kubernetes (kubectl)
    * Vagrant
    * Virtual Machines (Hyper-V, VMware, VirtualBox)
* **Development Tools and Languages:**
    * Installed Development Languages (Rust, Node.js, Perl, Ruby, PHP, Haskell)
    * Version Management Tools (nvm, rbenv, rustup, pyenv)
* **Debugging and Performance Monitoring:**
    * Load Averages, Memory Usage, IO Bottlenecks, GPU Utilization
    * Available Debugger Tools (lldb, gdb, strace, dtrace)
* **Version Control and CI/CD:**
    * Git Configuration and Remote Origins
    * CI/CD Pipeline Configuration Files (Common types)
* **Cloud and Remote Development:**
    * SSH Configurations and Active Remote Sessions
    * Cloud SDKs (AWS, GCP, Azure, DigitalOcean)
    * Remote Code Execution Environments (GitHub Codespaces, Gitpod)
* **Code Execution and Debugging:**
    * Active Debugger Sessions (Basic check)
    * Installed Debugging Tools (lldb, gdb, xdebug, pdb)
* **Build Systems and Dependency Management:**
    * Installed Build Tools (Make, CMake, Bazel, Ninja)
    * Detected Dependency Files (requirements.txt, package.json, Cargo.toml, etc.)
    * Installed Compilers (gcc, clang, javac)
* **Infrastructure and DevOps Tools:**
    * Local Kubernetes Configuration
    * DevOps Tools (Terraform, Pulumi)
    * Local Databases and Running Services (Simplified check for common DB services)
* **Testing and Quality Assurance:**
    * Installed Testing Frameworks (pytest, Jest, Mocha)
    * Code Linters and Formatters (flake8, pylint, eslint, prettier)
* **Machine Learning and AI Development:**
    * GPU and CUDA Information
    * PyTorch and TensorFlow Status (Installation and GPU availability)
* **Embedded Development / IoT:**
    * Installed Embedded SDKs (Arduino, ESP-IDF, Raspberry Pi Tools)
    * Connected Devices and Serial Ports (Simplified list of serial ports)
* **Productivity and Workflow Enhancements:**
    * Shell Aliases, Functions, and Custom Scripts
    * Shell History Analysis (Basic - last 20 lines of history)
    * Background Automation and Task Scheduling (Simplified check for cron/Scheduled Tasks)

## How it Works

This server is built using the Model Context Protocol (MCP) and operates as follows:

1.  **MCP Protocol:** It implements the MCP server protocol, allowing Cursor to communicate with it to discover and utilize its capabilities.
2.  **Stdio Transport:** The server uses the `stdio` transport, meaning it communicates with Cursor through standard input and output streams.
3.  **Information Gathering:** When Cursor's Agent requests information, this server executes various system commands (using `subprocess`) and Python libraries (`platform`, `os`, `sys`, `psutil`, `pyserial`, etc.) to collect data about your development environment.
4.  **Tool-Based Access:** Each information category is exposed as a tool within the MCP server. Cursor's Agent can then call these tools to retrieve specific pieces of information.
5.  **Markdown Output (Optional):** The server can optionally generate a Markdown file (`development_environment_info.md`) containing all the collected information for easier review and debugging.
6.  **Cursor Integration:** Cursor, acting as an MCP client, can connect to this server and automatically utilize the provided tools to enhance its understanding of your development context.

## Installation

To install and run this MCP server, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/carterlasalle/system_information_mcp.git
    cd system_information_mcp
    ```

2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    *   **On Linux/macOS:**
        ```bash
        source venv/bin/activate
        ```
    *   **On Windows:**
        ```bash
        venv\Scripts\activate
        ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration for Cursor

To connect this MCP server to Cursor, you need to configure it within Cursor's settings:

1.  **Open Cursor Settings:** Go to `Cursor Settings` > `Features` > `MCP`.
2.  **Add New MCP Server:** Click on the `+ Add New MCP Server` button.
3.  **Configure Server:** Fill in the form with the following details:
    *   **Type:** `stdio`
    *   **Name:** `DevEnvInfoServer` (or any name you prefer)
    *   **Command:**  Enter the command to run the server. If you are in the `system_information_mcp` directory and have activated the virtual environment, you can use:
        ```bash
        python claudemcp.py
        ```
        **Note:** If `python` is not in your system's PATH or you are using a specific Python executable, you may need to provide the full path to your Python interpreter followed by the path to `claudemcp.py`. For example:
        ```bash
        /path/to/your/python venv/bin/python claudemcp.py
        ```
4.  **Add Server:** Click the "Add Server" button.
5.  **Refresh Tool List (Optional):** You might need to manually press the refresh button in the top right corner of the MCP server list in Cursor to populate the tool list.

The server `DevEnvInfoServer` should now appear in your list of MCP servers in Cursor, and its tools should be available to the Agent in Composer.

## Usage in Cursor

Once configured, Cursor's Agent will automatically leverage the tools provided by `DevEnvInfoServer` when it deems them relevant to your requests.

*   **Automatic Tool Usage:**  When you interact with Cursor's Agent in Composer, it will intelligently decide if information about your development environment is needed to answer your questions or fulfill your requests. If so, it will automatically use the tools provided by this server in the background.
*   **Intentional Tool Prompting:** You can also explicitly instruct the Agent to use these tools by referring to them by name or description in your prompts. For example, you could ask:
    *   "What Python packages are installed in my current environment?"
    *   "List the available shells on my system using the DevEnvInfoServer tools."
*   **Tool Approval:** By default, Cursor will ask for your approval before executing any MCP tool. You can review the tool call arguments before approving.
*   **YOLO Mode (Optional):** If you prefer automatic tool execution without approval prompts, you can enable "YOLO Mode" in Cursor's MCP settings. Use this mode with caution, as it allows automatic execution of MCP tools.

Cursor will display the responses from the `DevEnvInfoServer` tools directly in the chat, providing you with the requested development environment information.
