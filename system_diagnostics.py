# -*- coding: utf-8 -*-
"""
ST-ACT Pipeline - System Diagnostics Utility
Checks available RAM, storage space, CPU capabilities, and Python environment
to evaluate suitability for processing large-scale anisotropic cell tracking datasets.
"""

import os
import sys
import shutil
import platform
import subprocess

# ANSI color codes for clean terminal outputs
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_ram_info():
    """
    Retrieves system RAM size in GB. Uses psutil if installed,
    falls back to Windows ctypes API, and then to CLI tools.
    """
    total_bytes = None
    available_bytes = None
    method = "fallback"

    try:
        import psutil
        mem = psutil.virtual_memory()
        total_bytes = mem.total
        available_bytes = mem.available
        method = "psutil"
    except ImportError:
        # Fallback for Windows (using native ctypes API - extremely robust!)
        if platform.system() == 'Windows':
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    total_bytes = stat.ullTotalPhys
                    available_bytes = stat.ullAvailPhys
                    method = "ctypes (Windows Native)"
            except Exception:
                pass
            
            # If ctypes fails, try wmic as second fallback
            if total_bytes is None:
                try:
                    out = subprocess.check_output(['wmic', 'ComputerSystem', 'get', 'TotalPhysicalMemory'], text=True)
                    lines = [line.strip() for line in out.split('\n') if line.strip()]
                    if len(lines) > 1 and lines[1].isdigit():
                        total_bytes = int(lines[1])
                    
                    out_free = subprocess.check_output(['wmic', 'OS', 'get', 'FreePhysicalMemory'], text=True)
                    lines_free = [line.strip() for line in out_free.split('\n') if line.strip()]
                    if len(lines_free) > 1 and lines_free[1].isdigit():
                        available_bytes = int(lines_free[1]) * 1024
                        method = "wmic"
                except Exception:
                    pass
        # Fallback for Linux
        elif platform.system() == 'Linux':
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemTotal:' in line:
                            total_bytes = int(line.split()[1]) * 1024
                        elif 'MemAvailable:' in line:
                            available_bytes = int(line.split()[1]) * 1024
                            method = "/proc/meminfo"
            except Exception:
                pass
        # Fallback for macOS
        elif platform.system() == 'Darwin':
            try:
                out = subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)
                total_bytes = int(out.strip())
                available_bytes = total_bytes // 2
                method = "sysctl"
            except Exception:
                pass

    total_gb = total_bytes / (1024**3) if total_bytes else None
    available_gb = available_bytes / (1024**3) if available_bytes else None
    return total_gb, available_gb, method

def run_diagnostics():
    # Enable ANSI terminal processing on Windows if running in cmd or powershell
    if platform.system() == 'Windows':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

    print(f"{Colors.HEADER}{Colors.BOLD}====================================================================={Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}           ST-ACT PIPELINE SYSTEM DIAGNOSTICS & HARDWARE AUDIT       {Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}====================================================================={Colors.ENDC}")
    print("This utility analyzes your PC configuration to verify if it can process")
    print("the massive 84GB cell-tracking volume safely without memory or storage crashes.\n")

    # 1. OS & Architecture Info
    print(f"{Colors.BOLD}[1/4] Operating System & Python Environment{Colors.ENDC}")
    os_name = platform.system()
    os_release = platform.release()
    os_arch = platform.machine()
    py_ver = sys.version.split()[0]
    is_64bit = sys.maxsize > 2**32

    print(f"  • Operating System: {os_name} ({os_release})")
    print(f"  • System Architecture: {os_arch}")
    print(f"  • Python Version: {py_ver} ({'64-bit' if is_64bit else '32-bit'})")
    
    if not is_64bit:
        print(f"  {Colors.FAIL}❌ WARNING: You are running 32-bit Python! It cannot address more than 4GB of RAM.{Colors.ENDC}")
        print(f"    Please install 64-bit Python to run the 84GB data pipeline.{Colors.ENDC}")
    else:
        print(f"  {Colors.GREEN}✔ Python architecture is 64-bit (Compatible){Colors.ENDC}")
    print()

    # 2. CPU info
    print(f"{Colors.BOLD}[2/4] CPU Capabilities{Colors.ENDC}")
    cpu_cores = os.cpu_count() or 1
    print(f"  • CPU Threads / Logical Cores: {cpu_cores}")
    if cpu_cores < 4:
        print(f"  {Colors.WARNING}⚠ Dynamic multithreading will be throttled. (Min 4 threads recommended){Colors.ENDC}")
    else:
        print(f"  {Colors.GREEN}✔ Core count is optimal for vectorized NumPy processing ({cpu_cores} threads available){Colors.ENDC}")
    print()

    # 3. RAM Info
    print(f"{Colors.BOLD}[3/4] System RAM (Random Access Memory){Colors.ENDC}")
    total_ram, avail_ram, ram_method = get_ram_info()
    
    if total_ram is not None:
        print(f"  • Total Installed RAM: {total_ram:.2f} GB")
        if avail_ram is not None:
            print(f"  • Currently Available RAM: {avail_ram:.2f} GB  (via {ram_method})")
        else:
            print(f"  • Currently Available RAM: [Failed to retrieve]")
        
        # RAM Assessment
        if total_ram < 8.0:
            print(f"  {Colors.FAIL}❌ DANGER: Your PC has only {total_ram:.1f}GB RAM. The 84GB dataset pipeline will CRASH due to Out-Of-Memory (OOM).{Colors.ENDC}")
            print(f"    {Colors.BOLD}Mitigation Required:{Colors.ENDC} You MUST lower the block-processing size. Set chunk_size below 32x64x64 in configuration.")
        elif total_ram < 15.5:
            print(f"  {Colors.WARNING}⚠ WARNING: Your PC has {total_ram:.1f}GB RAM. This is below the recommended 16GB minimum.{Colors.ENDC}")
            print(f"    {Colors.BOLD}Mitigation Recommended:{Colors.ENDC} Close intensive applications (Chrome, IDEs) before starting, or increase virtual memory paging size.")
        else:
            print(f"  {Colors.GREEN}✔ Installed RAM is sufficient ({total_ram:.1f} GB){Colors.ENDC}")
    else:
        print(f"  {Colors.WARNING}⚠ Could not verify total RAM. Ensure you have at least 16GB RAM installed.{Colors.ENDC}")
    print()

    # 4. Storage Info
    print(f"{Colors.BOLD}[4/4] Disk Space (Storage Capacity){Colors.ENDC}")
    try:
        total_disk, used_disk, free_disk = shutil.disk_usage(".")
        total_dg = total_disk / (1024**3)
        free_dg = free_disk / (1024**3)
        print(f"  • Target Drive: {os.path.abspath('.')[:3] if platform.system() == 'Windows' else 'Root (/) '}")
        print(f"  • Total Drive Size: {total_dg:.2f} GB")
        print(f"  • Available Free Space: {free_dg:.2f} GB")

        # Storage Assessment
        if free_dg < 100.0:
            print(f"  {Colors.FAIL}❌ STORAGE ALERT: You have only {free_dg:.1f} GB free space. The full 84GB dataset + output files require at least 100-120 GB.{Colors.ENDC}")
            print(f"    {Colors.BOLD}Mitigation Required:{Colors.ENDC} Free up storage space, or point 'store_path' to an external high-speed HDD/SSD drive.")
        elif free_dg < 180.0:
            print(f"  {Colors.WARNING}⚠ WARNING: You have {free_dg:.1f} GB free space. This is sufficient for the primary pipeline, but copy/cache operations may run low.{Colors.ENDC}")
        else:
            print(f"  {Colors.GREEN}✔ Available storage space is excellent ({free_dg:.1f} GB free){Colors.ENDC}")
    except Exception as e:
        print(f"  {Colors.WARNING}⚠ Could not read storage space statistics: {str(e)}{Colors.ENDC}")
    print()

    # Final Readiness Summary
    print(f"{Colors.BOLD}====================================================================={Colors.ENDC}")
    print(f"{Colors.BOLD}                         READINESS EVALUATION                        {Colors.ENDC}")
    print(f"{Colors.BOLD}====================================================================={Colors.ENDC}")
    
    is_ready = True
    recs = []

    if total_ram is not None and total_ram < 8.0:
        is_ready = False
        recs.append("Memory constraint: Allocate at least 16GB Virtual Memory (Pagefile) on Windows.")
    if 'free_dg' in locals() and free_dg < 100.0:
        is_ready = False
        recs.append("Storage constraint: Free up space until you have at least 100GB of free space on the drive.")
    if not is_64bit:
        is_ready = False
        recs.append("Environment constraint: Reinstall a 64-bit distribution of Python.")

    if is_ready:
        print(f"{Colors.GREEN}{Colors.BOLD}★ STATUS: SYSTEM IS FULLY COMPATIBLE!{Colors.ENDC}")
        print("Your PC meets the hardware specifications. The pipeline will execute with high-efficiency.")
    else:
        print(f"{Colors.WARNING}{Colors.BOLD}★ STATUS: SYSTEM MEETS CRITICAL BOTTLENECK LIMITS{Colors.ENDC}")
        print("To run the pipeline safely on this machine, apply these recommended adjustments:")
        for idx, rec in enumerate(recs, 1):
            print(f"  {idx}. {rec}")
    
    print(f"\n{Colors.BOLD}Tips for Maximum Performance & Stability:{Colors.ENDC}")
    print("  1. Close memory-hogging background applications (e.g. web browsers) before running the pipeline.")
    print("  2. Ensure your drive is an SSD (Solid State Drive) rather than an HDD, as Zarr's block-read speed")
    print("     is heavily bound by file I/O operations.")
    print("  3. Keep Blosc compression enabled (default). It maintains compressed blocks in cache, speeding up CPU.")
    print(f"{Colors.HEADER}{Colors.BOLD}====================================================================={Colors.ENDC}")

if __name__ == '__main__':
    run_diagnostics()