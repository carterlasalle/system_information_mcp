#!/usr/bin/env python3
"""
DevEnv MCP Server

A comprehensive Model Context Protocol (MCP) server that gathers detailed information
about the development environment to provide context for LLMs.
"""

import asyncio
import json
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import psutil
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

try:
    from mcp.server.fastmcp import FastMCP, Context
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mcp[cli]"])
    from mcp.server.fastmcp import FastMCP, Context

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("devenv-mcp")

# Define a class to track progress
@dataclass
class ProgressTracker:
    total: int = 0
    completed: int = 0
    current_task: str = ""
    
    def update(self, task: str, increment: int = 1) -> None:
        self.current_task = task
        self.completed += increment
        logger.info(f"Progress: [{self.completed}/{self.total}] {task}")
        
    def reset(self, total: int) -> None:
        self.total = total
        self.completed = 0
        self.current_task = ""

# Global progress tracker
progress = ProgressTracker()

# Define a context manager for task tracking
class TaskTracker:
    def __init__(self, task_name: str):
        self.task_name = task_name
        
    def __enter__(self):
        logger.info(f"Starting: {self.task_name}")
        progress.update(self.task_name)
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type:
            logger.error(f"Failed: {self.task_name} after {duration:.2f}s - {exc_val}")
        else:
            logger.info(f"Completed: {self.task_name} in {duration:.2f}s")
        return False  # Don't suppress exceptions

# Create a helper function to safely run commands
async def run_command(cmd: Union[str, List[str]], shell: bool = False, timeout: int = 30) -> Tuple[str, str, int]:
    """Run a command and return stdout, stderr, and return code."""
    if isinstance(cmd, str) and not shell:
        cmd = cmd.split()
    
    try:
        process = await asyncio.create_subprocess_shell(
            cmd if shell else " ".join(cmd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            shell=shell
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
            return stdout.decode('utf-8', errors='replace'), stderr.decode('utf-8', errors='replace'), process.returncode
        except asyncio.TimeoutError:
            try:
                process.kill()
            except:
                pass
            return "", f"Command timed out after {timeout} seconds", -1
    except Exception as e:
        return "", f"Failed to execute command: {str(e)}", -1

def safe_execute(func, default_value: Any = None) -> Any:
    """Execute a function and return a default value if it fails."""
    try:
        return func()
    except Exception as e:
        logger.warning(f"Error executing {func.__name__}: {str(e)}")
        return default_value

# Create the MCP server - remove the lifespan since we're simplifying
mcp = FastMCP("DevEnv Scanner")

# ----- System Information Collectors -----

async def collect_os_info() -> Dict[str, Any]:
    """Collect OS version information."""
    with TaskTracker("Collecting OS Information"):
        info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "architecture": platform.architecture(),
            "node": platform.node(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        }
        
        # Add distro information if on Linux
        if platform.system() == "Linux":
            try:
                # Try to get more detailed Linux distribution information
                stdout, stderr, returncode = await run_command("lsb_release -a", shell=True)
                if returncode == 0:
                    for line in stdout.splitlines():
                        if ":" in line:
                            key, value = line.split(":", 1)
                            info[f"distro_{key.strip().lower()}"] = value.strip()
                            
                # Try to read from os-release file
                if os.path.exists("/etc/os-release"):
                    with open("/etc/os-release", "r") as f:
                        for line in f:
                            if "=" in line:
                                key, value = line.split("=", 1)
                                info[f"os_{key.strip().lower()}"] = value.strip().strip('"')
            except Exception as e:
                logger.warning(f"Error collecting Linux distro info: {str(e)}")
        
        # Add macOS information if applicable
        if platform.system() == "Darwin":
            stdout, stderr, returncode = await run_command("sw_vers", shell=True)
            if returncode == 0:
                for line in stdout.splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        info[f"macos_{key.strip().lower().replace(' ', '_')}"] = value.strip()
        
        # Add Windows information if applicable
        if platform.system() == "Windows":
            try:
                stdout, stderr, returncode = await run_command("systeminfo", shell=True)
                if returncode == 0:
                    for line in stdout.splitlines():
                        if ":" in line:
                            key, value = line.split(":", 1)
                            info[f"win_{key.strip().lower().replace(' ', '_')}"] = value.strip()
            except Exception as e:
                logger.warning(f"Error collecting Windows system info: {str(e)}")
                
        return info

async def collect_hardware_info() -> Dict[str, Any]:
    """Collect hardware details."""
    with TaskTracker("Collecting Hardware Information"):
        info = {
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_freq": safe_execute(lambda: psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_partitions": safe_execute(lambda: [p._asdict() for p in psutil.disk_partitions()]),
            "disk_usage": {},
        }
        
        # Get disk usage for each partition
        for partition in psutil.disk_partitions():
            try:
                info["disk_usage"][partition.mountpoint] = psutil.disk_usage(partition.mountpoint)._asdict()
            except:
                pass
        
        # Try to get CPU model name
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            info["cpu_model"] = line.split(":", 1)[1].strip()
                            break
            except:
                pass
        elif platform.system() == "Darwin":
            stdout, stderr, returncode = await run_command("sysctl -n machdep.cpu.brand_string", shell=True)
            if returncode == 0:
                info["cpu_model"] = stdout.strip()
        elif platform.system() == "Windows":
            stdout, stderr, returncode = await run_command("wmic cpu get name", shell=True)
            if returncode == 0 and len(stdout.strip().split("\n")) > 1:
                info["cpu_model"] = stdout.strip().split("\n")[1].strip()
        
        return info

async def collect_python_versions() -> Dict[str, Any]:
    """Collect information about installed Python versions."""
    with TaskTracker("Collecting Python Versions"):
        info = {
            "current_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "current_executable": sys.executable,
            "implementation": platform.python_implementation(),
            "installed_versions": [],
        }
        
        # Try to find all Python versions
        python_executables = set()
        
        # Check common paths
        common_paths = os.environ.get("PATH", "").split(os.pathsep)
        for path in common_paths:
            for name in ["python", "python3"]:
                for version in range(6, 13):  # Python 3.6 to 3.12
                    for suffix in ["", ".exe"]:
                        executable = os.path.join(path, f"{name}{version}{suffix}")
                        if os.path.isfile(executable) and os.access(executable, os.X_OK):
                            python_executables.add(executable)
                
                # Also check for plain python/python3
                for suffix in ["", ".exe"]:
                    executable = os.path.join(path, f"{name}{suffix}")
                    if os.path.isfile(executable) and os.access(executable, os.X_OK):
                        python_executables.add(executable)
        
        # Check for pyenv installations
        pyenv_root = os.path.expanduser("~/.pyenv/versions")
        if os.path.isdir(pyenv_root):
            for version_dir in os.listdir(pyenv_root):
                for suffix in ["", ".exe"]:
                    executable = os.path.join(pyenv_root, version_dir, "bin", f"python{suffix}")
                    if os.path.isfile(executable) and os.access(executable, os.X_OK):
                        python_executables.add(executable)
        
        # Get version information for each Python executable
        for executable in python_executables:
            try:
                stdout, stderr, returncode = await run_command(
                    [executable, "-c", 
                     "import sys, platform; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro},{platform.python_implementation()},{sys.executable}')"
                    ]
                )
                if returncode == 0 and stdout.strip():
                    version, implementation, path = stdout.strip().split(",", 2)
                    info["installed_versions"].append({
                        "version": version,
                        "implementation": implementation,
                        "path": path,
                    })
            except Exception as e:
                logger.warning(f"Error checking Python version for {executable}: {str(e)}")
        
        return info

async def collect_network_config() -> Dict[str, Any]:
    """Collect network configuration information."""
    with TaskTracker("Collecting Network Configuration"):
        info = {
            "hostname": socket.gethostname(),
            "interfaces": safe_execute(lambda: [i._asdict() for i in psutil.net_if_addrs().values()], []),
            "connections": safe_execute(lambda: [c._asdict() for c in psutil.net_connections()], []),
        }
        
        # Get active interfaces
        try:
            info["active_interfaces"] = []
            for interface, addresses in psutil.net_if_addrs().items():
                for address in addresses:
                    if address.family == socket.AF_INET:  # IPv4
                        info["active_interfaces"].append({
                            "interface": interface,
                            "address": address.address,
                            "netmask": address.netmask,
                        })
        except Exception as e:
            logger.warning(f"Error collecting active interfaces: {str(e)}")
        
        # Try to get DNS configuration
        if platform.system() == "Linux":
            try:
                if os.path.exists("/etc/resolv.conf"):
                    with open("/etc/resolv.conf", "r") as f:
                        nameservers = []
                        for line in f:
                            if line.startswith("nameserver"):
                                nameservers.append(line.split()[1])
                        info["dns_servers"] = nameservers
            except Exception as e:
                logger.warning(f"Error collecting DNS information: {str(e)}")
        elif platform.system() == "Darwin":
            stdout, stderr, returncode = await run_command("scutil --dns", shell=True)
            if returncode == 0:
                nameservers = []
                for line in stdout.splitlines():
                    if "nameserver" in line and "[" in line and "]" in line:
                        server = line.split("[")[1].split("]")[0].strip()
                        if server:
                            nameservers.append(server)
                info["dns_servers"] = nameservers
        elif platform.system() == "Windows":
            stdout, stderr, returncode = await run_command("ipconfig /all", shell=True)
            if returncode == 0:
                nameservers = []
                for line in stdout.splitlines():
                    if "DNS Servers" in line and ":" in line:
                        server = line.split(":", 1)[1].strip()
                        if server:
                            nameservers.append(server)
                info["dns_servers"] = nameservers
        
        return info

async def collect_installed_programs() -> Dict[str, Any]:
    """Collect information about installed programs and package managers."""
    with TaskTracker("Collecting Installed Programs"):
        info = {
            "package_managers": {},
            "programs": [],
        }
        
        # Check common package managers
        package_managers = [
            ("brew", "brew --version"),
            ("npm", "npm --version"),
            ("pip", "pip --version"),
            ("pip3", "pip3 --version"),
            ("yarn", "yarn --version"),
            ("uv", "uv --version"),
            ("cargo", "cargo --version"),
            ("go", "go version"),
            ("apt", "apt --version"),
            ("yum", "yum --version"),
            ("dnf", "dnf --version"),
            ("pacman", "pacman --version"),
            ("conda", "conda --version"),
        ]
        
        for name, command in package_managers:
            stdout, stderr, returncode = await run_command(command, shell=True)
            if returncode == 0:
                info["package_managers"][name] = stdout.strip()
        
        # Find installed programs (platform specific)
        if platform.system() == "Windows":
            try:
                stdout, stderr, returncode = await run_command("wmic product get name,version", shell=True)
                if returncode == 0:
                    lines = stdout.strip().split("\n")[1:]  # Skip the header
                    for line in lines:
                        parts = line.strip().rsplit(" ", 1)
                        if len(parts) >= 2:
                            name, version = parts[0].strip(), parts[1].strip()
                            if name:
                                info["programs"].append({"name": name, "version": version})
            except Exception as e:
                logger.warning(f"Error collecting Windows installed programs: {str(e)}")
        
        elif platform.system() == "Darwin":
            try:
                stdout, stderr, returncode = await run_command("ls -la /Applications", shell=True)
                if returncode == 0:
                    for line in stdout.splitlines():
                        if ".app" in line:
                            parts = line.split()
                            if len(parts) >= 9:
                                name = " ".join(parts[8:])
                                info["programs"].append({"name": name, "version": "Unknown"})
                
                # Also check for Homebrew installed applications
                if "brew" in info["package_managers"]:
                    stdout, stderr, returncode = await run_command("brew list --versions", shell=True)
                    if returncode == 0:
                        for line in stdout.splitlines():
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                name, version = parts[0], " ".join(parts[1:])
                                info["programs"].append({"name": f"brew:{name}", "version": version})
            except Exception as e:
                logger.warning(f"Error collecting macOS installed programs: {str(e)}")
        
        elif platform.system() == "Linux":
            try:
                # Try to get dpkg installations (Debian/Ubuntu)
                stdout, stderr, returncode = await run_command("dpkg-query -l", shell=True)
                if returncode == 0:
                    lines = stdout.strip().split("\n")
                    for line in lines:
                        if line.startswith("ii"):
                            parts = line.split()
                            if len(parts) >= 3:
                                name, version = parts[1], parts[2]
                                info["programs"].append({"name": f"deb:{name}", "version": version})
                                
                # Try to get rpm installations (Fedora/CentOS/RHEL)
                stdout, stderr, returncode = await run_command("rpm -qa --queryformat '%{NAME} %{VERSION}\\n'", shell=True)
                if returncode == 0:
                    for line in stdout.splitlines():
                        if " " in line:
                            name, version = line.split(" ", 1)
                            info["programs"].append({"name": f"rpm:{name}", "version": version})
            except Exception as e:
                logger.warning(f"Error collecting Linux installed programs: {str(e)}")
        
        return info

async def collect_compilers() -> Dict[str, Any]:
    """Collect information about compilers and interpreters."""
    with TaskTracker("Collecting Compiler Information"):
        info = {
            "compilers": {},
            "interpreters": {},
        }
        
        # Check for common compilers
        compilers = [
            ("gcc", "gcc --version"),
            ("g++", "g++ --version"),
            ("clang", "clang --version"),
            ("clang++", "clang++ --version"),
            ("javac", "javac -version"),
            ("rustc", "rustc --version"),
            ("dotnet", "dotnet --version"),
            ("swiftc", "swiftc --version"),
        ]
        
        for name, command in compilers:
            stdout, stderr, returncode = await run_command(command, shell=True)
            if returncode == 0:
                output = stdout if stdout else stderr
                info["compilers"][name] = output.strip()
        
        # Check for common interpreters
        interpreters = [
            ("node", "node --version"),
            ("ruby", "ruby --version"),
            ("perl", "perl --version"),
            ("php", "php --version"),
            ("python", "python --version"),
            ("python3", "python3 --version"),
            ("R", "R --version"),
            ("julia", "julia --version"),
            ("lua", "lua -v"),
        ]
        
        for name, command in interpreters:
            stdout, stderr, returncode = await run_command(command, shell=True)
            if returncode == 0:
                output = stdout if stdout else stderr
                info["interpreters"][name] = output.strip()
        
        return info

async def collect_container_info() -> Dict[str, Any]:
    """Collect information about Docker, Kubernetes, etc."""
    with TaskTracker("Collecting Container Information"):
        info = {
            "docker": {
                "installed": False,
                "version": None,
                "containers": [],
                "images": [],
            },
            "kubernetes": {
                "installed": False,
                "version": None,
                "context": None,
            },
            "podman": {
                "installed": False,
                "version": None,
            },
        }
        
        # Check Docker
        docker_path = shutil.which("docker")
        if docker_path:
            info["docker"]["installed"] = True
            
            # Get Docker version
            stdout, stderr, returncode = await run_command("docker --version", shell=True)
            if returncode == 0:
                info["docker"]["version"] = stdout.strip()
            
            # Get running containers
            stdout, stderr, returncode = await run_command("docker ps", shell=True)
            if returncode == 0:
                lines = stdout.strip().split("\n")
                if len(lines) > 1:  # Skip header row
                    for line in lines[1:]:
                        info["docker"]["containers"].append(line)
            
            # Get images
            stdout, stderr, returncode = await run_command("docker images", shell=True)
            if returncode == 0:
                lines = stdout.strip().split("\n")
                if len(lines) > 1:  # Skip header row
                    for line in lines[1:]:
                        info["docker"]["images"].append(line)
        
        # Check Kubernetes
        kubectl_path = shutil.which("kubectl")
        if kubectl_path:
            info["kubernetes"]["installed"] = True
            
            # Get Kubernetes version
            stdout, stderr, returncode = await run_command("kubectl version --client", shell=True)
            if returncode == 0:
                info["kubernetes"]["version"] = stdout.strip()
            
            # Get current context
            stdout, stderr, returncode = await run_command("kubectl config current-context", shell=True)
            if returncode == 0:
                info["kubernetes"]["context"] = stdout.strip()
        
        # Check Podman
        podman_path = shutil.which("podman")
        if podman_path:
            info["podman"]["installed"] = True
            
            # Get Podman version
            stdout, stderr, returncode = await run_command("podman --version", shell=True)
            if returncode == 0:
                info["podman"]["version"] = stdout.strip()
        
        return info

async def collect_gpu_info() -> Dict[str, Any]:
    """Collect information about GPUs and CUDA."""
    with TaskTracker("Collecting GPU Information"):
        info = {
            "gpus": [],
            "cuda": {
                "installed": False,
                "version": None,
            },
            "opencl": {
                "installed": False,
                "version": None,
            },
        }
        
        # Platform-specific GPU detection
        if platform.system() == "Windows":
            try:
                stdout, stderr, returncode = await run_command("wmic path win32_VideoController get name,driverversion,videoprocessor", shell=True)
                if returncode == 0:
                    lines = stdout.strip().split("\n")
                    if len(lines) > 1:  # Skip header row
                        for line in lines[1:]:
                            parts = line.strip().split()
                            if parts:
                                info["gpus"].append({
                                    "name": " ".join(parts[:-2]) if len(parts) > 2 else " ".join(parts),
                                    "driver_version": parts[-2] if len(parts) > 2 else None,
                                    "processor": parts[-1] if len(parts) > 1 else None,
                                })
            except Exception as e:
                logger.warning(f"Error collecting Windows GPU information: {str(e)}")
        
        elif platform.system() == "Linux":
            try:
                # Try to get information from lspci
                stdout, stderr, returncode = await run_command("lspci | grep -i 'vga\\|3d\\|display'", shell=True)
                if returncode == 0:
                    for line in stdout.splitlines():
                        if ":" in line:
                            info["gpus"].append({"name": line.split(":", 1)[1].strip()})
                
                # Try to get NVIDIA-specific information
                stdout, stderr, returncode = await run_command("nvidia-smi -L", shell=True)
                if returncode == 0:
                    for line in stdout.splitlines():
                        if "GPU " in line:
                            info["gpus"].append({"name": line.strip(), "type": "nvidia"})
            except Exception as e:
                logger.warning(f"Error collecting Linux GPU information: {str(e)}")
        
        elif platform.system() == "Darwin":
            try:
                stdout, stderr, returncode = await run_command("system_profiler SPDisplaysDataType", shell=True)
                if returncode == 0:
                    current_gpu = None
                    for line in stdout.splitlines():
                        line = line.strip()
                        if "Chipset Model" in line:
                            current_gpu = {"name": line.split(":", 1)[1].strip()}
                            info["gpus"].append(current_gpu)
                        elif current_gpu and ":" in line:
                            key, value = line.split(":", 1)
                            current_gpu[key.strip().lower().replace(" ", "_")] = value.strip()
            except Exception as e:
                logger.warning(f"Error collecting macOS GPU information: {str(e)}")
        
        # Check for CUDA
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            info["cuda"]["installed"] = True
            
            # Get CUDA version
            stdout, stderr, returncode = await run_command("nvcc --version", shell=True)
            if returncode == 0:
                for line in stdout.splitlines():
                    if "release" in line:
                        info["cuda"]["version"] = line.strip()
        
        # Check for nvidia-smi
        nvidia_smi_path = shutil.which("nvidia-smi")
        if nvidia_smi_path:
            stdout, stderr, returncode = await run_command("nvidia-smi", shell=True)
            if returncode == 0:
                info["cuda"]["nvidia_smi"] = stdout.strip()
        
        return info

async def collect_dev_processes() -> Dict[str, Any]:
    """Collect information about running development processes."""
    with TaskTracker("Collecting Development Processes"):
        info = {
            "dev_processes": [],
            "editors": [],
            "servers": [],
            "databases": [],
        }
        
        # Define process patterns to look for
        dev_patterns = [
            # Editors
            r"code", r"vscode", r"atom", r"sublime_text", r"vim", r"nvim", r"gvim",
            r"emacs", r"pycharm", r"intellij", r"webstorm", r"phpstorm", r"goland",
            
            # Servers
            r"nginx", r"apache", r"httpd", r"node", r"flask", r"django", r"gunicorn",
            r"uwsgi", r"webpack", r"vite", r"parcel", r"nodemon", r"pm2",
            
            # Databases
            r"mysql", r"postgres", r"mongodb", r"redis", r"sqlite", r"mariadb",
            
            # Dev tools
            r"npm", r"pip", r"conda", r"jupyter", r"docker", r"kubectl", r"java",
            r"go", r"python", r"ruby", r"perl", r"php", r"rustc", r"cargo",
        ]
        
        try:
            # Get all running processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username']):
                try:
                    process_info = proc.info
                    process_name = process_info['name'].lower() if process_info['name'] else ""
                    
                    # Convert cmdline to a string for pattern matching
                    cmdline = " ".join(process_info['cmdline']).lower() if process_info['cmdline'] else ""
                    
                    # Check if this matches any dev process pattern
                    for pattern in dev_patterns:
                        if re.search(pattern, process_name) or re.search(pattern, cmdline):
                            info["dev_processes"].append({
                                "pid": process_info['pid'],
                                "name": process_info['name'],
                                "cmdline": process_info['cmdline'],
                                "username": process_info['username'],
                            })
                            
                            # Categorize the process
                            if any(x in process_name or x in cmdline for x in ["code", "vscode", "atom", "sublime", "vim", "nvim", "emacs", "pycharm", "intellij"]):
                                info["editors"].append(process_info['name'])
                            elif any(x in process_name or x in cmdline for x in ["nginx", "apache", "httpd", "node", "flask", "django", "gunicorn"]):
                                info["servers"].append(process_info['name'])
                            elif any(x in process_name or x in cmdline for x in ["mysql", "postgres", "mongodb", "redis", "sqlite"]):
                                info["databases"].append(process_info['name'])
                            
                            # Don't add duplicates
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        except Exception as e:
            logger.warning(f"Error collecting development processes: {str(e)}")
        
        # Deduplicate
        info["editors"] = list(set(info["editors"]))
        info["servers"] = list(set(info["servers"]))
        info["databases"] = list(set(info["databases"]))
        
        return info

# ----- Tool Endpoints -----

@mcp.tool()
async def get_system_info(ctx: Context, category: str) -> str:
    """
    Get detailed information about a specific category of system information.

    Args:
        category: Category of information to retrieve. Available categories:
                 os, hardware, python, network, programs, compilers, containers, gpu, processes
    """
    # Define collector functions for each category
    category_collectors = {
    "os": collect_os_info,
    "hardware": collect_hardware_info,
    "python": collect_python_versions,
    "network": collect_network_config,
    "programs": collect_installed_programs,
    "compilers": collect_compilers,
    "containers": collect_container_info,
    "gpu": collect_gpu_info,
    "processes": collect_dev_processes
    }
    
    if category not in category_collectors:
        return f"Error: Invalid category '{category}'. Available categories: {', '.join(category_collectors.keys())}"
    
    # Collect the data directly (no caching for simplicity)
    with TaskTracker(f"Collecting {category} information"):
        result = await category_collectors[category]()
    
    return json.dumps(result, indent=2)

@mcp.tool()
async def analyze_cursor_environment(ctx: Context) -> str:
    """
    Analyze the development environment specifically focusing on relevant components for Cursor IDE.
    This provides a targeted analysis of components that affect code editing, execution, and AI assistance.

    Returns:
        str: A JSON string containing:
            - OS and hardware capabilities for running Cursor
            - Python environment details and package compatibility 
            - GPU availability for AI model acceleration
            - Compiler/build tools for code execution
            - System resource metrics impacting performance
    """
    # List of important categories to scan for Cursor
    categories = ["os", "hardware", "python"]
    
    # Run a system scan for these categories
    with TaskTracker("Analyzing Cursor environment"):
        # Initialize collectors
        collectors = {
            "os": collect_os_info,
            "hardware": collect_hardware_info,
            "python": collect_python_versions,
            "compilers": collect_compilers,
            "gpu": collect_gpu_info
        }
        
        # Collect data concurrently
        results = {}
        futures = []
        
        for category in categories:
            if category in collectors:
                futures.append((category, asyncio.create_task(collectors[category]())))
        
        # Collect results
        for category, future in futures:
            try:
                with TaskTracker(f"Collecting {category} information"):
                    results[category] = await future
            except Exception as e:
                logger.error(f"Error collecting {category} information: {str(e)}")
                results[category] = {"error": str(e)}
    
    # Analyze the environment and provide insights
    insights = []
    
    # OS insights
    if "os" in results:
        os_name = results["os"].get("system", "Unknown")
        insights.append(f"**Operating System**: {os_name} {results['os'].get('release', '')}")
    
    # Hardware insights
    if "hardware" in results:
        hw = results["hardware"]
        insights.append(f"**CPU**: {hw.get('cpu_count_logical', 'Unknown')} cores")
        if "memory_total" in hw:
            ram_gb = hw["memory_total"] / (1024**3)
            insights.append(f"**RAM**: {ram_gb:.1f} GB")
    
    # Python insights
    if "python" in results:
        py_info = results["python"]
        insights.append(f"**Python**: {py_info.get('current_version', 'Unknown')} ({py_info.get('implementation', 'Unknown')})")
    
    # Format insights
    markdown = "# Cursor Development Environment Analysis\n\n"
    markdown += f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    markdown += "## Environment Summary\n\n"
    for insight in insights:
        markdown += f"- {insight}\n"
    
    return markdown

@mcp.tool()
async def scan_system(ctx: Context, categories: Union[List[str], str, None] = None, output_format: Optional[str] = "markdown") -> str:
    """
    Scan the system and collect information about the specified categories.

    Args:
        categories: Categories to scan. Can be:
                   - A single category as a string (e.g., "hardware")
                   - A list of categories (e.g., ["os", "hardware"])
                   - None to scan all categories
                   Available categories: os, hardware, python, network, programs, compilers, containers, gpu, processes
        output_format: Format for the output. Supported values: "markdown"(DONT USE), "json"(USE THIS). Default is "json". ONLY USE JSON FOR NOW

    Examples:
        To scan OS only: "os" or ["os"]
        To scan multiple categories: ["os", "hardware", "python"]
        To scan all categories: leave empty or use null
    """
    # Define all available categories
    all_categories = {
        "os": collect_os_info,
        "hardware": collect_hardware_info,
        "python": collect_python_versions,
        "network": collect_network_config,
        "programs": collect_installed_programs,
        "compilers": collect_compilers,
        "containers": collect_container_info,
        "gpu": collect_gpu_info,
        "processes": collect_dev_processes
    }
    
    # Handle string input for categories by converting to list
    if isinstance(categories, str):
        categories = [categories]
    
    # If no categories specified, scan all
    if not categories:
        categories = list(all_categories.keys())
    
    # Initialize progress tracker
    progress.reset(len(categories))
    
    # Collect information for each category
    results = {}
    futures = []

    # Start async tasks
    with TaskTracker("Starting System Scan"):
        for category in categories:
            if category in all_categories:
                futures.append((category, asyncio.create_task(all_categories[category]())))
    
    # Collect results
    for category, future in futures:
        try:
            with TaskTracker(f"Collecting {category} information"):
                results[category] = await future
        except Exception as e:
            logger.error(f"Error collecting {category} information: {str(e)}")
            results[category] = {"error": str(e)}
    
    # Format the output based on requested format
    if output_format and output_format.lower() == "json":
        return json.dumps(results, indent=2)
    else:
        # Create a detailed markdown report
        markdown = "# System Scan Report\n\n"
        markdown += f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add system summary
        if "os" in results:
            os_info = results["os"]
            markdown += "## System Summary\n\n"
            markdown += f"**OS**: {os_info.get('system', 'Unknown')} {os_info.get('release', '')}\n"
            markdown += f"**Version**: {os_info.get('version', 'Unknown')}\n"
            markdown += f"**Architecture**: {os_info.get('machine', 'Unknown')}\n"
            markdown += f"**Hostname**: {os_info.get('node', 'Unknown')}\n\n"
        
        # Add hardware summary
        if "hardware" in results:
            hw_info = results["hardware"]
            markdown += "## Hardware\n\n"
            if "cpu_model" in hw_info:
                markdown += f"**CPU**: {hw_info.get('cpu_model', 'Unknown')}\n"
            markdown += f"**CPU Cores**: {hw_info.get('cpu_count_physical', 'Unknown')} physical, {hw_info.get('cpu_count_logical', 'Unknown')} logical\n"
            
            if "memory_total" in hw_info:
                memory_gb = hw_info.get("memory_total", 0) / (1024 * 1024 * 1024)
                markdown += f"**Memory**: {memory_gb:.2f} GB total\n\n"
        
            # Display disk information
            if "disk_partitions" in hw_info and hw_info["disk_partitions"]:
                markdown += "### Storage\n\n"
                markdown += "| Mount Point | Type | Size |\n"
                markdown += "|------------|------|------|\n"
                
                for partition in hw_info["disk_partitions"]:
                    mountpoint = partition.get("mountpoint", "Unknown")
                    if "disk_usage" in hw_info and mountpoint in hw_info["disk_usage"]:
                        usage = hw_info["disk_usage"][mountpoint]
                        size_gb = usage.get("total", 0) / (1024 * 1024 * 1024)
                        markdown += f"| {mountpoint} | {partition.get('fstype', 'Unknown')} | {size_gb:.2f} GB |\n"
                
                markdown += "\n"
        
        # Add Python information
        if "python" in results:
            py_info = results["python"]
            markdown += "## Python Environment\n\n"
            markdown += f"**Current Python**: {py_info.get('current_version', 'Unknown')} ({py_info.get('implementation', 'Unknown')})\n"
            markdown += f"**Executable**: {py_info.get('current_executable', 'Unknown')}\n\n"
            
            if "installed_versions" in py_info and py_info["installed_versions"]:
                markdown += "### Installed Python Versions\n\n"
                markdown += "| Version | Implementation | Path |\n"
                markdown += "|---------|---------------|------|\n"
                
                for version in py_info["installed_versions"]:
                    markdown += f"| {version.get('version', 'Unknown')} | {version.get('implementation', 'Unknown')} | {version.get('path', 'Unknown')} |\n"
                
                markdown += "\n"
        
        # Add Network information
        if "network" in results:
            net_info = results["network"]
            markdown += "## Network Configuration\n\n"
            markdown += f"**Hostname**: {net_info.get('hostname', 'Unknown')}\n\n"
            
            if "active_interfaces" in net_info and net_info["active_interfaces"]:
                markdown += "### Network Interfaces\n\n"
                markdown += "| Interface | IP Address | Netmask |\n"
                markdown += "|-----------|------------|--------|\n"
                
                for interface in net_info["active_interfaces"]:
                    markdown += f"| {interface.get('interface', 'Unknown')} | {interface.get('address', 'Unknown')} | {interface.get('netmask', 'Unknown')} |\n"
                
                markdown += "\n"
            
            if "dns_servers" in net_info and net_info["dns_servers"]:
                markdown += "### DNS Servers\n\n"
                for dns in net_info["dns_servers"]:
                    markdown += f"- {dns}\n"
                
                markdown += "\n"
        
        # Add compiler/interpreter information
        if "compilers" in results:
            comp_info = results["compilers"]
            markdown += "## Development Environment\n\n"
            
            if "compilers" in comp_info and comp_info["compilers"]:
                markdown += "### Compilers\n\n"
                for name, version in comp_info["compilers"].items():
                    markdown += f"- **{name}**: {version.splitlines()[0] if version.splitlines() else version}\n"
                markdown += "\n"
            
            if "interpreters" in comp_info and comp_info["interpreters"]:
                markdown += "### Interpreters\n\n"
                for name, version in comp_info["interpreters"].items():
                    markdown += f"- **{name}**: {version.splitlines()[0] if version.splitlines() else version}\n"
                markdown += "\n"
        
        # Add container information
        if "containers" in results:
            container_info = results["containers"]
            markdown += "## Containerization\n\n"
            
            if container_info.get("docker", {}).get("installed", False):
                markdown += f"**Docker**: {container_info['docker'].get('version', 'Installed')}\n"
                
                if container_info["docker"].get("containers"):
                    markdown += f"- Running containers: {len(container_info['docker']['containers'])}\n"
                
                if container_info["docker"].get("images"):
                    markdown += f"- Available images: {len(container_info['docker']['images'])}\n"
                
                markdown += "\n"
            
            if container_info.get("kubernetes", {}).get("installed", False):
                markdown += f"**Kubernetes**: {container_info['kubernetes'].get('version', 'Installed')}\n"
                markdown += f"- Current context: {container_info['kubernetes'].get('context', 'Unknown')}\n\n"
        
        # Add GPU information
        if "gpu" in results:
            gpu_info = results["gpu"]
            markdown += "## GPU Information\n\n"
            
            if gpu_info.get("gpus"):
                markdown += "### GPUs\n\n"
                for i, gpu in enumerate(gpu_info["gpus"]):
                    markdown += f"**GPU {i+1}**: {gpu.get('name', 'Unknown')}\n"
                    
                    # Add driver version if available
                    if "driver_version" in gpu and gpu["driver_version"]:
                        markdown += f"- Driver version: {gpu['driver_version']}\n"
                
                markdown += "\n"
            
            if gpu_info.get("cuda", {}).get("installed", False):
                markdown += f"**CUDA**: {gpu_info['cuda'].get('version', 'Installed')}\n\n"
        
        # Add running processes if available
        if "processes" in results:
            proc_info = results["processes"]
            markdown += "## Development Processes\n\n"
            
            if proc_info.get("editors"):
                markdown += "**Active Editors**: " + ", ".join(proc_info["editors"]) + "\n"
            
            if proc_info.get("servers"):
                markdown += "**Active Servers**: " + ", ".join(proc_info["servers"]) + "\n"
            
            if proc_info.get("databases"):
                markdown += "**Active Databases**: " + ", ".join(proc_info["databases"]) + "\n"
        
        return markdown
# Add these tool functions to your script to replace the resources
#EEEEEEE
@mcp.tool()
async def get_os_info(ctx: Context) -> str:
    """
    Get detailed information about the operating system.
    Returns formatted details about the OS, version, architecture, etc.
    """
    result = await collect_os_info()
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_hardware_info(ctx: Context) -> str:
    """
    Get detailed information about the system hardware.
    Returns CPU, memory, disk information in formatted JSON.
    """
    result = await collect_hardware_info()
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_python_info(ctx: Context) -> str:
    """
    Get detailed information about Python installations.
    Returns current Python version and all detected Python environments.
    """
    result = await collect_python_versions()
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_network_config(ctx: Context) -> str:
    """
    Get detailed information about network configuration.
    Returns network interfaces, connections, and DNS configuration.
    """
    # Assuming you kept the collect_network_config function in your script
    result = await collect_network_config()
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_installed_programs(ctx: Context) -> str:
    """
    Get detailed information about installed programs and package managers.
    Returns list of detected apps and package managers on the system.
    """
    # Assuming you kept the collect_installed_programs function
    result = await collect_installed_programs()
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_dev_tools_info(ctx: Context) -> str:
    """
    Get detailed information about development tools.
    Returns installed compilers, IDEs, and related development software.
    """
    # This requires collect_compilers and other functions
    results = {}
    
    # Collect compilers data
    with TaskTracker("Collecting development tools information"):
        try:
            results["compilers"] = await collect_compilers()
        except Exception as e:
            logger.error(f"Error collecting compiler information: {str(e)}")
            results["compilers"] = {"error": str(e)}
            
    return json.dumps(results, indent=2)

@mcp.tool()
async def get_container_info(ctx: Context) -> str:
    """
    Get detailed information about containerization tools.
    Returns Docker, Kubernetes, and other container-related information.
    """
    # Assuming you kept the collect_container_info function
    result = await collect_container_info()
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_gpu_info(ctx: Context) -> str:
    """
    Get detailed information about GPUs and CUDA.
    Returns information about installed graphics cards and CUDA capabilities.
    """
    # Assuming you kept the collect_gpu_info function
    result = await collect_gpu_info()
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_running_processes(ctx: Context) -> str:
    """
    Get information about currently running development processes.
    Returns information about editors, servers, and other dev-related processes.
    """
    # Assuming you kept the collect_dev_processes function
    result = await collect_dev_processes()
    return json.dumps(result, indent=2)

@mcp.tool()
async def generate_system_report(ctx: Context, output_path: Optional[str] = None, format: str = "markdown") -> str:
    """
    Generate a comprehensive system report with all available information.
    
    Args:
        output_path: Optional path to save the report. If not provided, returns the report directly.
        format: Format for the output. Supported values: "markdown"(DONT USE), "json"(USE) (DONT USE MARKDOWN, USE JSON FOR NOW)
    """
    # List all categories to scan
    categories = ["os", "hardware", "python", "network", "programs", "compilers", 
                  "gpu", "containers", "processes"]
    
    # Use the existing scan_system function
    report = await scan_system(ctx, categories, format)
    
    # Write to file if path provided
    if output_path:
        try:
            with open(os.path.expanduser(output_path), "w") as f:
                f.write(report)
            return f"System report saved to {output_path} ({len(report)} bytes)"
        except Exception as e:
            return f"Error writing report to file: {str(e)}\n\n{report}"
    
    # Otherwise return directly
    return report

# Run the server directly with stdio transport
if __name__ == "__main__":
    mcp.run(transport='stdio')